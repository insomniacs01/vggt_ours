# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from .dpt_head import DPTHead
from .track_modules.base_track_predictor import BaseTrackerPredictor


class TrackHead(nn.Module):
    """
    跟踪头，使用DPT头处理令牌，并使用BaseTrackerPredictor进行跟踪。
    跟踪是迭代执行的，通过多次迭代优化预测。
    """

    def __init__(
        self,
        dim_in,
        patch_size=14,
        features=128,
        iters=4,
        predict_conf=True,
        stride=2,
        corr_levels=7,
        corr_radius=4,
        hidden_size=384,
    ):
        """
        初始化TrackHead模块。

        参数：
            dim_in (int): 来自主干的令牌输入维度。
            patch_size (int): vision transformer中使用的图像patch大小。
            features (int): 特征提取器输出中的特征通道数。
            iters (int): 跟踪预测的优化迭代次数。
            predict_conf (bool): 是否为跟踪点预测置信度分数。
            stride (int): 跟踪器预测器的步幅值。
            corr_levels (int): 相关金字塔层级数
            corr_radius (int): 相关计算的半径，控制搜索区域。
            hidden_size (int): 跟踪器网络中隐藏层的大小。
        """
        super().__init__()

        self.patch_size = patch_size

        # 基于DPT架构的特征提取器
        # 将令牌处理成用于跟踪的特征图
        self.feature_extractor = DPTHead(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            feature_only=True,  # 仅输出特征，不进行激活
            down_ratio=2,  # 将空间维度减少2倍
            pos_embed=False,
        )

        # 预测点轨迹的跟踪器模块
        # 获取特征图并预测坐标和可见性
        self.tracker = BaseTrackerPredictor(
            latent_dim=features,  # 匹配特征提取器的output_dim
            predict_conf=predict_conf,
            stride=stride,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
            hidden_size=hidden_size,
        )

        self.iters = iters

    def forward(self, aggregated_tokens_list, images, patch_start_idx, query_points=None, iters=None):
        """
        TrackHead的前向传播。

        参数：
            aggregated_tokens_list (list): 来自主干的聚合令牌列表。
            images (torch.Tensor): 输入图像，形状为(B, S, C, H, W)，其中：
                                   B = 批大小，S = 序列长度。
            patch_start_idx (int): patch令牌的起始索引。
            query_points (torch.Tensor, optional): 要跟踪的初始查询点。
                                                  如果为None，点由跟踪器初始化。
            iters (int, optional): 优化迭代次数。如果为None，使用self.iters。

        返回：
            tuple:
                - coord_preds (torch.Tensor): 跟踪点的预测坐标。
                - vis_scores (torch.Tensor): 跟踪点的可见性分数。
                - conf_scores (torch.Tensor): 跟踪点的置信度分数（如果predict_conf=True）。
        """
        B, S, _, H, W = images.shape

        # 从令牌中提取特征
        # feature_maps的形状为(B, S, C, H//2, W//2)，因为down_ratio=2
        feature_maps = self.feature_extractor(aggregated_tokens_list, images, patch_start_idx)

        # 如果未指定，使用默认迭代次数
        if iters is None:
            iters = self.iters

        # 使用提取的特征执行跟踪
        coord_preds, vis_scores, conf_scores = self.tracker(query_points=query_points, fmaps=feature_maps, iters=iters)

        return coord_preds, vis_scores, conf_scores