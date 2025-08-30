# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from vggt.models.aggregator import Aggregator
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead
from vggt.heads.detection_head import DetectionHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(self, img_size=518, patch_size=14, embed_dim=1024,
                 enable_camera=True, enable_point=True, enable_depth=True,
                 enable_track=True, enable_detection=False,
                 num_classes=10,
                 # DN-DETR和DAB-DETR参数
                 use_dn=True, use_dab=True):
        super().__init__()

        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = DPTHead(dim_in=2 * embed_dim, output_dim=4, activation="inv_log",
                                  conf_activation="expp1") if enable_point else None
        self.depth_head = DPTHead(dim_in=2 * embed_dim, output_dim=2, activation="exp",
                                  conf_activation="expp1") if enable_depth else None
        self.track_head = TrackHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_track else None

        # 增强的3D检测头
        self.detection_head = DetectionHead(
            dim_in=2 * embed_dim,
            patch_size=patch_size,
            num_classes=num_classes,
            max_per_image=300,
            training_sample_size=100,
            conf_threshold=0.01,
            use_dn=use_dn,  # DN-DETR
            use_dab=use_dab,  # DAB-DETR
            dn_number=100,
            dn_noise_scale=0.4
        ) if enable_detection else None

    def forward(self, images: torch.Tensor, query_points: torch.Tensor = None,
                gt_boxes: torch.Tensor = None, gt_classes: torch.Tensor = None,
                epoch: int = 0):
        """
        VGGT模型的前向传播。

        参数:
            images: 输入图像 [S, 3, H, W] 或 [B, S, 3, H, W]
            query_points: 查询点坐标 [N, 2] 或 [B, N, 2]
            gt_boxes: GT 3D框 (训练时用于DN) [B, M, 7]
            gt_classes: GT类别 (训练时用于DN) [B, M]
            epoch: 当前训练epoch（用于渐进式训练）

        返回:
            dict: 包含各种预测结果的字典
        """
        # 添加批量维度
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        aggregated_tokens_list, patch_start_idx = self.aggregator(images)

        predictions = {}

        with torch.cuda.amp.autocast(enabled=False):
            if self.detection_head is not None:
                # 传递epoch信息给detection head
                detection_results = self.detection_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    gt_boxes=gt_boxes if self.training else None,
                    gt_classes=gt_classes if self.training else None,
                    epoch=epoch if self.training else 0  # 传递epoch
                )
                predictions.update(detection_results)

            # 其他head的处理保持不变...
            if self.camera_head is not None:
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list

            if self.depth_head is not None:
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.point_head is not None:
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf

        if self.track_head is not None and query_points is not None:
            track_list, vis, conf = self.track_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx, query_points=query_points
            )
            predictions["track"] = track_list[-1]
            predictions["vis"] = vis
            predictions["conf"] = conf

        if not self.training:
            predictions["images"] = images

        return predictions