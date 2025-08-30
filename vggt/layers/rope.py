# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.


# 2D旋转位置嵌入（RoPE）的实现。

# 该模块提供了2D旋转位置嵌入的简洁实现，
# 将原始的RoPE概念扩展到处理2D空间位置。

# 灵感来源：
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class PositionGetter:
    """为网格中的patch生成和缓存2D空间位置。

    该类高效地管理2D网格中patch的空间坐标生成，
    缓存结果以避免冗余计算。

    属性：
        position_cache: 存储不同网格维度的预计算位置张量的字典。
    """

    def __init__(self):
        """使用空缓存初始化位置生成器。"""
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """为一批patch生成空间位置。

        参数：
            batch_size: 批次中的样本数。
            height: 以patch为单位的网格高度。
            width: 以patch为单位的网格宽度。
            device: 位置张量的目标设备。

        返回：
            形状为(batch_size, height*width, 2)的张量，包含网格中每个位置的y,x坐标，
            为每个批次项重复。
        """
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D旋转位置嵌入实现。

    该模块基于输入令牌的2D空间位置对其应用旋转位置嵌入。
    它分别处理垂直和水平维度的位置相关特征旋转。

    参数：
        frequency: 位置嵌入的基础频率。默认：100.0
        scaling_factor: 频率计算的缩放因子。默认：1.0

    属性：
        base_frequency: 计算位置嵌入的基础频率。
        scaling_factor: 缩放计算频率的因子。
        frequency_cache: 存储预计算频率分量的缓存。
    """

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        """初始化2D RoPE模块。"""
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算旋转嵌入的频率分量。

        参数：
            dim: 特征维度（必须为偶数）。
            seq_len: 最大序列长度。
            device: 计算的目标设备。
            dtype: 计算张量的数据类型。

        返回：
            频率分量的(余弦, 正弦)张量元组。
        """
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            # 计算频率带
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            # 生成位置相关的频率
            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            # 计算并缓存频率分量
            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        """通过分割和重组特征维度来执行特征旋转。

        参数：
            x: 要旋转的输入张量。

        返回：
            旋转后的特征张量。
        """
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self, tokens: torch.Tensor, positions: torch.Tensor, cos_comp: torch.Tensor, sin_comp: torch.Tensor
    ) -> torch.Tensor:
        """沿一个维度应用1D旋转位置嵌入。

        参数：
            tokens: 输入令牌特征。
            positions: 位置索引。
            cos_comp: 旋转的余弦分量。
            sin_comp: 旋转的正弦分量。

        返回：
            应用了旋转位置嵌入的令牌。
        """
        # 使用频率分量嵌入位置
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]

        # 应用旋转
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """对输入令牌应用2D旋转位置嵌入。

        参数：
            tokens: 形状为(batch_size, n_heads, n_tokens, dim)的输入张量。
                   特征维度(dim)必须能被4整除。
            positions: 形状为(batch_size, n_tokens, 2)的位置张量，包含
                      每个令牌的y和x坐标。

        返回：
            与输入形状相同的张量，应用了2D旋转位置嵌入。

        异常：
            AssertionError: 如果输入维度无效或位置格式错误。
        """
        # 验证输入
        assert tokens.size(-1) % 2 == 0, "特征维度必须为偶数"
        assert positions.ndim == 3 and positions.shape[-1] == 2, "位置必须具有形状(batch_size, n_tokens, 2)"

        # 计算每个空间方向的特征维度
        feature_dim = tokens.size(-1) // 2

        # 获取频率分量
        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(feature_dim, max_position, tokens.device, tokens.dtype)

        # 分割特征用于垂直和水平处理
        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        # 分别为每个维度应用RoPE
        vertical_features = self._apply_1d_rope(vertical_features, positions[..., 0], cos_comp, sin_comp)
        horizontal_features = self._apply_1d_rope(horizontal_features, positions[..., 1], cos_comp, sin_comp)

        # 组合处理后的特征
        return torch.cat((vertical_features, horizontal_features), dim=-1)