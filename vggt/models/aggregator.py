# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    Aggregator应用交替注意力于输入帧，
    如VGGT: Visual Geometry Grounded Transformer中所述。

    记得设置model.train()以启用梯度检查点来减少内存使用。

    参数:
        img_size (int): 图像大小（像素）。
        patch_size (int): PatchEmbed的每个补丁大小。
        embed_dim (int): 令牌嵌入的维度。
        depth (int): 块的数量。
        num_heads (int): 注意力头的数量。
        mlp_ratio (float): MLP隐藏维度与嵌入维度的比率。
        num_register_tokens (int): 寄存器令牌的数量。
        block_fn (nn.Module): 用于注意力的块类型（默认为Block）。
        qkv_bias (bool): 是否在QKV投影中包含偏置。
        proj_bias (bool): 是否在输出投影中包含偏置。
        ffn_bias (bool): 是否在MLP层中包含偏置。
        patch_embed (str): 补丁嵌入的类型。例如"conv"或"dinov2_vitl14_reg"。
        aa_order (list[str]): 交替注意力的顺序，例如["frame", "global"]。
        aa_block_size (int): 在切换之前将多少块分组在每个注意力类型下。如果不必要，设置为1。
        qk_norm (bool): 是否应用QK归一化。
        rope_freq (int): 旋转嵌入的基础频率。-1表示禁用。
        init_values (float): 层缩放的初始缩放值。
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
    ):
        super().__init__()

        self.__build_patch_embed__(patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim)

        # 如果频率>0，初始化旋转位置嵌入
        self.rope = RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        self.position_getter = PositionGetter() if self.rope is not None else None

        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # 验证depth是否可被aa_block_size整除
        if self.depth % self.aa_block_size != 0:
            raise ValueError(f"depth ({depth}) 必须可被 aa_block_size ({aa_block_size}) 整除")

        self.aa_block_num = self.depth // self.aa_block_size

        # 注意：我们有两个相机令牌，一个用于第一帧，一个用于其余帧
        # 寄存器令牌也是如此
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.randn(1, 2, num_register_tokens, embed_dim))

        # 补丁令牌从相机和寄存器令牌之后开始
        self.patch_start_idx = 1 + num_register_tokens

        # 用小值初始化参数
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # 将归一化常数注册为缓冲区
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.use_reentrant = False # 硬编码为False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        构建补丁嵌入层。如果是'conv'，我们使用
        简单的PatchEmbed卷积层。否则，我们使用视觉变换器。
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=3, embed_dim=embed_dim)
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # 禁用掩码令牌的梯度更新
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> Tuple[List[torch.Tensor], int]:
        """
        参数:
            images (torch.Tensor): 输入图像，形状为 [B, S, 3, H, W]，范围为 [0, 1]。
                B: 批量大小, S: 序列长度, 3: RGB通道, H: 高度, W: 宽度

        返回:
            (list[torch.Tensor], int):
                注意力块的输出列表，
                以及指示补丁令牌开始位置的patch_start_idx。
        """
        B, S, C_in, H, W = images.shape

        if C_in != 3:
            raise ValueError(f"期望3个输入通道，得到 {C_in}")

        # 归一化图像并重塑以进行补丁嵌入
        images = (images - self._resnet_mean) / self._resnet_std

        # 重塑为 [B*S, C, H, W] 以进行补丁嵌入
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)

        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        _, P, C = patch_tokens.shape

        # 扩展相机和寄存器令牌以匹配批量大小和序列长度
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)

        # 将特殊令牌与补丁令牌连接
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # 不要为特殊令牌（相机和寄存器令牌）使用位置嵌入
            # 所以将特殊令牌的pos设置为0
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # 更新P，因为我们添加了特殊令牌
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for block_i in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"未知的注意力类型: {attn_type}")

            for i in range(len(frame_intermediates)):
                # 连接帧和全局中间结果，[B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None):
        """
        处理帧注意力块。我们保持令牌形状为 (B*S, P, C)。
        """
        # 如果需要，重塑令牌或位置：
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        # 默认情况下，self.aa_block_size=1，一次处理一个块
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.frame_blocks[frame_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(self, tokens, B, S, P, C, global_idx, pos=None):
        """
        处理全局注意力块。我们保持令牌形状为 (B, S*P, C)。
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = []

        # 默认情况下，self.aa_block_size=1，一次处理一个块
        for _ in range(self.aa_block_size):
            if self.training:
                tokens = checkpoint(self.global_blocks[global_idx], tokens, pos, use_reentrant=self.use_reentrant)
            else:
                tokens = self.global_blocks[global_idx](tokens, pos=pos)
            global_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates


def slice_expand_and_flatten(token_tensor, B, S):
    """
    处理形状为 (1, 2, X, C) 的专用令牌以进行多帧处理：
    1) 仅为第一帧使用第一个位置（索引=0）
    2) 为所有剩余帧（S-1帧）使用第二个位置（索引=1）
    3) 扩展两者以匹配批量大小B
    4) 连接形成 (B, S, X, C)，其中每个序列有1个第一位置令牌
       后跟(S-1)个第二位置令牌
    5) 展平为 (B*S, X, C) 以进行处理

    返回:
        torch.Tensor: 处理后的令牌，形状为 (B*S, X, C)
    """

    # 切出"查询"令牌 => 形状 (1, 1, ...)
    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # 切出"其他"令牌 => 形状 (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # 连接 => 形状 (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # 最后展平 => 形状 (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined