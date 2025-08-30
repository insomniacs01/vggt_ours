# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch import nn, einsum
from einops import rearrange, repeat
from einops.layers.torch import Rearrange, Reduce

from hydra.utils import instantiate
from omegaconf import OmegaConf

from .track_modules.track_refine import refine_track
from .track_modules.blocks import BasicEncoder, ShallowEncoder
from .track_modules.base_track_predictor import BaseTrackerPredictor


class TrackerPredictor(nn.Module):
    def __init__(self, **extra_args):
        super(TrackerPredictor, self).__init__()
        """
        初始化跟踪器预测器。

        coarse_predictor和fine_predictor都构建为BaseTrackerPredictor，
        查看track_modules/base_track_predictor.py

        coarse_fnet和fine_fnet都构建为2D CNN网络
        查看track_modules/blocks.py中的BasicEncoder和ShallowEncoder
        """
        # 定义粗略预测器配置
        coarse_stride = 4
        self.coarse_down_ratio = 2

        # 直接创建网络而不使用instantiate
        self.coarse_fnet = BasicEncoder(stride=coarse_stride)
        self.coarse_predictor = BaseTrackerPredictor(stride=coarse_stride)

        # 创建步长为1的精细预测器
        self.fine_fnet = ShallowEncoder(stride=1)
        self.fine_predictor = BaseTrackerPredictor(
            stride=1,
            depth=4,
            corr_levels=3,
            corr_radius=3,
            latent_dim=32,
            hidden_size=256,
            fine=True,
            use_spaceatt=False,
        )

    def forward(
        self, images, query_points, fmaps=None, coarse_iters=6, inference=True, fine_tracking=True, fine_chunk=40960
    ):
        """
        参数:
            images (torch.Tensor): RGB图像，范围[0, 1]，形状为B x S x 3 x H x W。
            query_points (torch.Tensor): 查询点的2D xy坐标，相对于左上角，形状为B x N x 2。
            fmaps (torch.Tensor, optional): 预计算的特征图。默认为None。
            coarse_iters (int, optional): 粗略预测的迭代次数。默认为6。
            inference (bool, optional): 是否执行推理。默认为True。
            fine_tracking (bool, optional): 是否执行精细跟踪。默认为True。

        返回:
            tuple: 包含fine_pred_track、coarse_pred_track、pred_vis和pred_score的元组。
        """

        if fmaps is None:
            batch_num, frame_num, image_dim, height, width = images.shape
            reshaped_image = images.reshape(batch_num * frame_num, image_dim, height, width)
            fmaps = self.process_images_to_fmaps(reshaped_image)
            fmaps = fmaps.reshape(batch_num, frame_num, -1, fmaps.shape[-2], fmaps.shape[-1])

            if inference:
                torch.cuda.empty_cache()

        # 粗略预测
        coarse_pred_track_lists, pred_vis = self.coarse_predictor(
            query_points=query_points, fmaps=fmaps, iters=coarse_iters, down_ratio=self.coarse_down_ratio
        )
        coarse_pred_track = coarse_pred_track_lists[-1]

        if inference:
            torch.cuda.empty_cache()

        if fine_tracking:
            # 细化粗略预测
            fine_pred_track, pred_score = refine_track(
                images, self.fine_fnet, self.fine_predictor, coarse_pred_track, compute_score=False, chunk=fine_chunk
            )

            if inference:
                torch.cuda.empty_cache()
        else:
            fine_pred_track = coarse_pred_track
            pred_score = torch.ones_like(pred_vis)

        return fine_pred_track, coarse_pred_track, pred_vis, pred_score

    def process_images_to_fmaps(self, images):
        """
        此函数处理图像以进行推理。

        参数:
            images (torch.Tensor): 要处理的图像，形状为S x 3 x H x W。

        返回:
            torch.Tensor: 处理后的特征图。
        """
        if self.coarse_down_ratio > 1:
            # 是否缩小输入图像以节省内存
            fmaps = self.coarse_fnet(
                F.interpolate(images, scale_factor=1 / self.coarse_down_ratio, mode="bilinear", align_corners=True)
            )
        else:
            fmaps = self.coarse_fnet(images)

        return fmaps