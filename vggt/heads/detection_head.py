import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np

from .dpt_head import DPTHead, _make_scratch, _make_fusion_block, custom_interpolate
from .utils import create_uv_grid, position_grid_to_embed


class DetectionHead(nn.Module):

    def __init__(
            self,
            dim_in: int,
            patch_size: int = 14,
            num_classes: int = 10,
            features: int = 256,
            out_channels: List[int] = [256, 512, 1024, 1024],
            intermediate_layer_idx: List[int] = [4, 11, 17, 23],
            pos_embed: bool = True,
            conf_threshold: float = 0.01,
            max_per_image: int = 300,
            training_sample_size: int = 100,
            # DN-DETR参数
            use_dn: bool = True,
            dn_number: int = 100,
            dn_noise_scale: float = 0.4,
            dn_label_noise_ratio: float = 0.5,
            # DAB-DETR参数
            use_dab: bool = True,
            num_queries: int = 300,
            # 新增：密集采样参数
            dense_sample_ratio: float = 0.3,  # 密集采样比例
            initial_obj_bias: float = 0.01,  # objectness初始偏置
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_classes = num_classes
        self.pos_embed = pos_embed
        self.intermediate_layer_idx = intermediate_layer_idx
        self.conf_threshold = conf_threshold
        self.max_per_image = max_per_image
        self.training_sample_size = training_sample_size
        self.dense_sample_ratio = dense_sample_ratio

        # DN-DETR参数
        self.use_dn = use_dn
        self.dn_number = dn_number
        self.dn_noise_scale = dn_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio

        # DAB-DETR参数
        self.use_dab = use_dab
        self.num_queries = num_queries

        # 输出维度: 7(3D box) + num_classes + 1(objectness)
        self.output_dim = 7 + num_classes + 1

        # 基础层
        self.norm = nn.LayerNorm(dim_in)

        self.projects = nn.ModuleList([
            nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0)
            for oc in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0),
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0),
            nn.Identity(),
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
        ])

        # 特征融合
        self.scratch = _make_scratch(out_channels, features, expand=False)
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        # DAB-DETR: 可学习的3D anchor boxes
        if self.use_dab:
            self.anchor_boxes = nn.Embedding(num_queries, 7)
            nn.init.uniform_(self.anchor_boxes.weight, -1.0, 1.0)
            self.anchor_embed = nn.Sequential(
                nn.Linear(7, features // 4),
                nn.ReLU(inplace=True),
                nn.Linear(features // 4, features)
            )

        # 检测头
        self.detection_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(features // 2, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.output_dim, kernel_size=1)
        )

        # DN组件
        if self.use_dn:
            self.dn_embed = nn.Linear(features, features)

        # 改进的初始化：设置objectness偏置为正值
        self._initialize_detection_head(initial_obj_bias)

    def _initialize_detection_head(self, obj_bias: float = 0.01):
        """初始化检测头，特别是objectness层"""
        # 获取最后一个卷积层
        last_conv = self.detection_head[-1]

        # 初始化权重
        nn.init.xavier_uniform_(last_conv.weight)

        # 设置偏置
        if last_conv.bias is not None:
            nn.init.zeros_(last_conv.bias)
            # 设置objectness通道的偏置为正值，鼓励初期预测更多正样本
            last_conv.bias.data[-1] = obj_bias

            # 设置box regression的合理初始值
            last_conv.bias.data[3:6] = torch.log(torch.tensor([2.0, 1.0, 1.5]))  # 默认尺寸

    def forward(
            self,
            aggregated_tokens_list: List[torch.Tensor],
            images: torch.Tensor,
            patch_start_idx: int,
            gt_boxes: Optional[List[torch.Tensor]] = None,
            gt_classes: Optional[List[torch.Tensor]] = None,
            epoch: int = 0,  # 新增：当前epoch
    ) -> Dict[str, torch.Tensor]:
        B, S, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # 提取DPT风格的特征
        features = self._extract_features(
            aggregated_tokens_list, B, S, patch_h, patch_w,
            patch_start_idx, H, W
        )

        # 融合特征
        fused = self._fuse_features(features)

        # DN-DETR: 训练时准备去噪queries
        dn_meta = None
        if self.training and self.use_dn and gt_boxes is not None:
            dn_queries, dn_meta = self._prepare_dn_queries(
                gt_boxes, gt_classes, B, S, fused.device
            )
            if dn_queries is not None:
                fused = self._merge_dn_features(fused, dn_queries)

        # DAB-DETR: 使用anchor boxes调制特征
        if self.use_dab:
            fused = self._apply_dab_features(fused, B, S)

        # 上采样到原始分辨率
        fused = custom_interpolate(
            fused, (H, W), mode="bilinear", align_corners=True
        )

        # 生成检测输出
        raw_output = self.detection_head(fused)

        # 后处理
        if self.training:
            detections = self._postprocess_training(raw_output, B, S, H, W, epoch)
            if dn_meta is not None:
                detections['dn_meta'] = dn_meta
            detections['epoch'] = epoch  # 传递epoch信息
        else:
            detections = self._postprocess_inference(raw_output, B, S, H, W)

        return detections

    def _postprocess_training(self, raw_output, B, S, H, W, epoch=0):
        """训练时的后处理，支持渐进式采样策略"""
        device = raw_output.device

        # 激活函数处理（保持不变）
        positions = raw_output[:, :3].permute(0, 2, 3, 1)
        positions = torch.tanh(positions) * torch.tensor([10, 5, 2], device=device)
        sizes = raw_output[:, 3:6].permute(0, 2, 3, 1)
        sizes = torch.exp(sizes.clamp(min=-3, max=3))
        angles = raw_output[:, 6:7].permute(0, 2, 3, 1)
        angles = torch.tanh(angles) * np.pi
        boxes = torch.cat([positions, sizes, angles], dim=-1)
        classes = raw_output[:, 7:7 + self.num_classes].permute(0, 2, 3, 1)
        objectness = torch.sigmoid(raw_output[:, -1])

        # 重塑
        boxes = boxes.view(B, S, H * W, 7)
        classes = classes.view(B, S, H * W, self.num_classes)
        objectness = objectness.view(B, S, H * W)

        # 渐进式采样策略
        sampled_indices = []
        sampled_boxes = []
        sampled_classes = []
        sampled_scores = []

        for b in range(B):
            for s in range(S):
                if epoch < 5:
                    # 早期训练：使用密集网格采样
                    indices = self._dense_grid_sampling(H, W, self.training_sample_size, device)
                elif epoch < 10:
                    # 中期训练：混合采样（修复版本）
                    n_grid = int(self.training_sample_size * self.dense_sample_ratio)
                    n_score = self.training_sample_size - n_grid

                    grid_idx = self._dense_grid_sampling(H, W, n_grid, device)

                    scores = objectness[b, s]
                    top_scores, score_idx = scores.topk(min(n_score * 2, H * W - n_grid))  # 采样更多候选

                    # 确保不重复
                    mask = torch.ones(H * W, dtype=torch.bool, device=device)
                    mask[grid_idx] = False
                    valid_score_idx = score_idx[mask[score_idx]]

                    # 确保正好n_score个样本
                    if len(valid_score_idx) >= n_score:
                        score_idx = valid_score_idx[:n_score]
                    else:
                        # 如果不够，用随机采样补充
                        num_needed = n_score - len(valid_score_idx)
                        all_indices = torch.arange(H * W, device=device)
                        mask[score_idx] = False  # 排除已选择的
                        available = all_indices[mask]
                        if len(available) > 0:
                            extra = available[torch.randperm(len(available), device=device)[:num_needed]]
                            score_idx = torch.cat([valid_score_idx, extra])
                        else:
                            # 如果还是不够，重复已有的
                            score_idx = torch.cat([
                                valid_score_idx,
                                valid_score_idx[torch.randint(0, len(valid_score_idx), (num_needed,), device=device)]
                            ])

                    indices = torch.cat([grid_idx, score_idx])
                else:
                    # 后期训练：主要基于分数采样
                    scores = objectness[b, s]
                    n_top = self.training_sample_size // 2
                    top_scores, top_idx = scores.topk(min(n_top, H * W))

                    # 确保正好n_top个
                    if len(top_idx) < n_top:
                        # 用随机索引补充
                        extra_needed = n_top - len(top_idx)
                        extra = torch.randint(0, H * W, (extra_needed,), device=device)
                        top_idx = torch.cat([top_idx, extra])

                    n_random = self.training_sample_size - n_top
                    remaining_idx = torch.randperm(H * W, device=device)
                    remaining_idx = remaining_idx[~torch.isin(remaining_idx, top_idx)][:n_random]

                    # 如果remaining不够，补充
                    if len(remaining_idx) < n_random:
                        extra_needed = n_random - len(remaining_idx)
                        extra = torch.randint(0, H * W, (extra_needed,), device=device)
                        remaining_idx = torch.cat([remaining_idx, extra])

                    indices = torch.cat([top_idx, remaining_idx])

                # 最终确保正好是training_sample_size个样本
                if len(indices) != self.training_sample_size:
                    if len(indices) > self.training_sample_size:
                        indices = indices[:self.training_sample_size]
                    else:
                        # 补充到目标数量
                        extra_needed = self.training_sample_size - len(indices)
                        extra = indices[torch.randint(0, len(indices), (extra_needed,), device=device)]
                        indices = torch.cat([indices, extra])

                sampled_indices.append(indices)
                sampled_boxes.append(boxes[b, s, indices])
                sampled_classes.append(classes[b, s, indices])
                sampled_scores.append(objectness[b, s, indices])

        # 现在所有采样都有相同的大小，可以安全堆叠
        return {
            'pred_boxes': torch.stack([torch.stack(sampled_boxes[i * S:(i + 1) * S])
                                       for i in range(B)]),
            'pred_classes': torch.stack([torch.stack(sampled_classes[i * S:(i + 1) * S])
                                         for i in range(B)]),
            'pred_scores': torch.stack([torch.stack(sampled_scores[i * S:(i + 1) * S])
                                        for i in range(B)]),
            'pred_indices': sampled_indices,
            'dense_features': {
                'boxes': boxes,
                'classes': classes,
                'objectness': objectness
            }
        }

    def _dense_grid_sampling(self, H: int, W: int, n_samples: int, device: torch.device) -> torch.Tensor:
        """密集网格采样，确保空间覆盖"""
        total = H * W

        if n_samples >= total:
            return torch.arange(total, device=device)

        # 计算步长以均匀采样
        stride = int(np.sqrt(total / n_samples))
        stride = max(1, stride)

        # 生成网格点
        h_indices = torch.arange(0, H, stride, device=device)
        w_indices = torch.arange(0, W, stride, device=device)

        # 创建网格
        grid_h, grid_w = torch.meshgrid(h_indices, w_indices, indexing='ij')
        indices = grid_h.flatten() * W + grid_w.flatten()

        # 如果不够，随机补充
        if len(indices) < n_samples:
            remaining = n_samples - len(indices)
            extra = torch.randperm(total, device=device)
            extra = extra[~torch.isin(extra, indices)][:remaining]
            indices = torch.cat([indices, extra])
        elif len(indices) > n_samples:
            # 如果太多，随机选择
            perm = torch.randperm(len(indices), device=device)[:n_samples]
            indices = indices[perm]

        return indices

    # 其他方法保持不变...
    def _apply_dab_features(self, fused: torch.Tensor, B: int, S: int) -> torch.Tensor:
        """应用DAB-DETR的anchor特征"""
        batch_size, channels, height, width = fused.shape

        # 生成anchor特征
        anchor_embeddings = self.anchor_embed(self.anchor_boxes.weight)

        # 通过全局平均池化应用anchor信息
        anchor_weights = torch.mean(anchor_embeddings, dim=0, keepdim=True)
        anchor_weights = anchor_weights.view(1, channels, 1, 1)
        anchor_weights = torch.sigmoid(anchor_weights)

        fused = fused * (1 + 0.1 * anchor_weights.expand_as(fused))

        return fused

    # 保留其他原有方法...
    def _extract_features(self, aggregated_tokens_list, B, S, patch_h, patch_w,
                          patch_start_idx, H, W):
        """提取多尺度特征"""
        features = []

        for idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            x = x.reshape(B * S, -1, x.shape[-1])
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape((B * S, -1, patch_h, patch_w))
            x = self.projects[idx](x)

            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)

            x = self.resize_layers[idx](x)
            features.append(x)

        return features

    def _fuse_features(self, features):
        """融合多尺度特征"""
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        out = self.scratch.refinenet1(out, layer_1_rn)

        return out

    def _prepare_dn_queries(self, gt_boxes, gt_classes, B, S, device):
        """准备DN queries"""
        if not gt_boxes or all(len(b) == 0 for b in gt_boxes):
            return None, None

        dn_queries_list = []
        dn_meta = {'dn_positive_idx': [], 'dn_num_group': self.dn_number}

        for b in range(B):
            if len(gt_boxes[b]) == 0:
                continue

            gt = gt_boxes[b].to(device)
            num_gt = len(gt)

            # 重复GT以创建多个噪声版本
            dn_group_gt = gt.repeat(self.dn_number, 1)

            # 添加噪声
            if self.dn_noise_scale > 0:
                pos_noise = torch.randn((self.dn_number * num_gt, 3), device=device)
                pos_noise = pos_noise * self.dn_noise_scale
                dn_group_gt[:, :3] = dn_group_gt[:, :3] + pos_noise

                size_noise = torch.randn((self.dn_number * num_gt, 3), device=device)
                size_noise = size_noise * self.dn_noise_scale * 0.5
                dn_group_gt[:, 3:6] = dn_group_gt[:, 3:6] * torch.exp(size_noise)

                angle_noise = torch.randn((self.dn_number * num_gt, 1), device=device)
                angle_noise = angle_noise * self.dn_noise_scale
                dn_group_gt[:, 6:7] = dn_group_gt[:, 6:7] + angle_noise

            dn_queries_list.append(dn_group_gt)
            dn_meta['dn_positive_idx'].append(list(range(num_gt * self.dn_number)))

        if dn_queries_list:
            max_num = max(q.shape[0] for q in dn_queries_list)
            padded_queries = []
            for q in dn_queries_list:
                if q.shape[0] < max_num:
                    padding = torch.zeros((max_num - q.shape[0], 7), device=device)
                    q = torch.cat([q, padding], dim=0)
                padded_queries.append(q)
            dn_queries = torch.stack(padded_queries, dim=0)
            return dn_queries, dn_meta

        return None, None

    def _merge_dn_features(self, fused, dn_queries):
        """合并DN features"""
        B = dn_queries.shape[0]
        dn_feat = self.dn_embed(dn_queries.mean(dim=1, keepdim=True))
        dn_feat = dn_feat.permute(0, 2, 1).unsqueeze(-1)

        if dn_feat.shape[0] < fused.shape[0]:
            dn_feat = dn_feat.repeat(fused.shape[0] // B, 1, 1, 1)

        fused = fused + 0.1 * dn_feat.expand(-1, -1, fused.shape[2], fused.shape[3])
        return fused

    def _postprocess_inference(self, raw_output, B, S, H, W):
        """推理时的后处理"""
        device = raw_output.device

        positions = raw_output[:, :3].permute(0, 2, 3, 1)
        positions = torch.tanh(positions) * torch.tensor([10, 5, 2], device=device)

        sizes = raw_output[:, 3:6].permute(0, 2, 3, 1)
        sizes = torch.exp(sizes.clamp(min=-3, max=3))

        angles = raw_output[:, 6:7].permute(0, 2, 3, 1)
        angles = torch.tanh(angles) * np.pi

        boxes = torch.cat([positions, sizes, angles], dim=-1)

        classes = raw_output[:, 7:7 + self.num_classes].permute(0, 2, 3, 1)
        class_probs = F.softmax(classes, dim=-1)

        objectness = torch.sigmoid(raw_output[:, -1])

        boxes = boxes.view(B, S, H * W, 7)
        class_probs = class_probs.view(B, S, H * W, self.num_classes)
        objectness = objectness.view(B, S, H * W)

        # 收集有效检测
        detections = []
        for b in range(B):
            batch_dets = []
            for s in range(S):
                valid_mask = objectness[b, s] > self.conf_threshold
                valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(1)

                if len(valid_indices) == 0:
                    batch_dets.append({
                        'boxes': torch.empty((0, 7), device=device),
                        'scores': torch.empty((0,), device=device),
                        'classes': torch.empty((0, self.num_classes), device=device),
                    })
                    continue

                if len(valid_indices) > self.max_per_image:
                    scores = objectness[b, s, valid_indices]
                    top_scores, top_idx = scores.topk(self.max_per_image)
                    valid_indices = valid_indices[top_idx]

                batch_dets.append({
                    'boxes': boxes[b, s, valid_indices],
                    'scores': objectness[b, s, valid_indices],
                    'classes': class_probs[b, s, valid_indices],
                })

            detections.append(batch_dets)

        return {'detections': detections}

    def _apply_pos_embed(self, x, W, H, ratio=0.1):
        """应用位置编码"""
        patch_w, patch_h = x.shape[-1], x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H,
                                   dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed