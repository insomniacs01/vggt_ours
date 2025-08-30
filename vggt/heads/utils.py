# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    将2D位置网格(HxWx2)转换为正弦嵌入(HxWxC)

    参数：
        pos_grid: 形状为(H, W, 2)的张量，包含2D坐标
        embed_dim: 嵌入的输出通道维度

    返回：
        形状为(H, W, embed_dim)的位置嵌入张量
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # 展平为(H*W, 2)

    # 分别处理x和y坐标
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # 组合并重塑
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]


def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    该函数使用正弦和余弦函数从给定网格生成1D位置嵌入。

    参数：
    - embed_dim: 嵌入维度。
    - pos: 生成嵌入的位置。

    返回：
    - emb: 生成的1D位置嵌入。
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), 外积

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()


# 灵感来源于 https://github.com/microsoft/moge


def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    创建形状为(width, height, 2)的归一化UV网格。

    网格根据宽高比在水平和垂直方向上展开，
    确保左上角在(-x_span, -y_span)，右下角
    在(x_span, y_span)，通过平面的对角线进行归一化。

    参数：
        width (int): 水平点数。
        height (int): 垂直点数。
        aspect_ratio (float, optional): 宽高比。默认为width/height。
        dtype (torch.dtype, optional): 结果张量的数据类型。
        device (torch.device, optional): 创建张量的设备。

    返回：
        torch.Tensor: UV坐标的(width, height, 2)张量。
    """
    # 如果未明确提供，则推导宽高比
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # 计算X和Y的归一化跨度
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # 建立线性空间边界
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # 生成1D坐标
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # 创建2D网格(width x height)并堆叠成UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid