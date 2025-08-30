import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np


class DetectionLoss(nn.Module):
    """
    融合DN-DETR和MS-DETR思想的3D检测损失，支持渐进式训练
    """

    def __init__(self, num_classes=10, focal_alpha=0.25, focal_gamma=2.0,
                 iou_loss_type='diou', max_gt=50,
                 # 损失权重
                 pos_weight=5.0,
                 size_weight=2.0,
                 angle_weight=1.0,
                 cls_weight=1.0,
                 obj_weight=0.5,
                 iou_weight=2.0,
                 center_weight=2.0,
                 dn_loss_weight=1.0,
                 # 新增：未匹配GT惩罚
                 unmatched_gt_weight=5.0,
                 # 渐进式训练参数
                 warmup_epochs=5):
        super().__init__()
        self.num_classes = num_classes
        self.max_gt = max_gt
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.iou_loss_type = iou_loss_type
        self.dn_loss_weight = dn_loss_weight
        self.pos_weight = pos_weight
        self.unmatched_gt_weight = unmatched_gt_weight
        self.warmup_epochs = warmup_epochs

        self.loss_weights = nn.ParameterDict({
            'center': nn.Parameter(torch.tensor(center_weight)),
            'size': nn.Parameter(torch.tensor(size_weight)),
            'angle': nn.Parameter(torch.tensor(angle_weight)),
            'cls': nn.Parameter(torch.tensor(cls_weight)),
            'obj': nn.Parameter(torch.tensor(obj_weight)),
            'iou': nn.Parameter(torch.tensor(iou_weight)),
            'dn': nn.Parameter(torch.tensor(dn_loss_weight)),
            'unmatched': nn.Parameter(torch.tensor(unmatched_gt_weight))
        })

    def forward(self, pred: Dict, batch: Dict, epoch: int = 0) -> Dict[str, torch.Tensor]:
        device = pred['pred_boxes'].device
        B, S, N, _ = pred['pred_boxes'].shape

        # 验证并准备GT数据
        gt_boxes, gt_classes = self._prepare_and_validate_gt(batch, B, device)

        # 渐进式匹配策略
        assignments, unmatched_gts = self.progressive_matching(
            pred, gt_boxes, gt_classes, epoch
        )

        losses = {}

        # 基础损失（始终计算）
        losses['loss_det_center'] = self._center_loss(pred, gt_boxes, assignments) * self.loss_weights['center']
        losses['loss_det_obj'] = self._objectness_loss_progressive(pred, assignments, epoch) * self.loss_weights['obj']

        # 渐进式添加其他损失
        if epoch >= self.warmup_epochs:
            losses['loss_det_size'] = self._size_loss(pred, gt_boxes, assignments) * self.loss_weights['size']
            losses['loss_det_angle'] = self._angle_loss(pred, gt_boxes, assignments) * self.loss_weights['angle']
            losses['loss_det_cls'] = self._class_loss_focal(pred, gt_classes, assignments) * self.loss_weights['cls']

        if epoch >= self.warmup_epochs * 2:
            losses['loss_det_iou'] = self._diou_loss(pred, gt_boxes, assignments) * self.loss_weights['iou']

        # 未匹配GT的额外损失（鼓励更多预测框去学习）
        if unmatched_gts is not None and epoch < self.warmup_epochs * 2:
            losses['loss_unmatched_gt'] = self._unmatched_gt_loss(
                pred, unmatched_gts, device
            ) * self.loss_weights['unmatched']

        # DN损失
        if 'dn_meta' in pred and pred['dn_meta'] is not None:
            dn_loss = self._compute_dn_loss(pred, gt_boxes, gt_classes, pred['dn_meta'])
            losses['loss_det_dn'] = dn_loss * self.loss_weights['dn']

        losses['loss_detection'] = sum(losses.values())
        return losses

    def progressive_matching(self, pred: Dict, gt_boxes: List, gt_classes: List, epoch: int):
        """渐进式匹配策略，确保早期有足够的正样本"""
        B, S, N, _ = pred['pred_boxes'].shape
        device = pred['pred_boxes'].device
        assignments = torch.full((B, S, N), -1, dtype=torch.long, device=device)

        # 记录未匹配的GT
        all_unmatched_gts = []

        # 动态调整匹配参数
        if epoch < self.warmup_epochs:
            k_factor = 10  # 早期：每个GT匹配更多预测框
            use_objectness = False  # 早期：不依赖objectness
            distance_threshold = 5.0  # 早期：更大的匹配半径
        elif epoch < self.warmup_epochs * 2:
            k_factor = 5
            use_objectness = True
            distance_threshold = 3.0
        else:
            k_factor = 3
            use_objectness = True
            distance_threshold = 2.0

        for b in range(B):
            if b >= len(gt_boxes) or len(gt_boxes[b]) == 0:
                continue

            for s in range(S):
                n_gt = len(gt_boxes[b])
                if n_gt == 0:
                    continue

                pred_box = pred['pred_boxes'][b, s]
                pred_cls = pred['pred_classes'][b, s]
                pred_obj = pred['pred_scores'][b, s]

                # 计算成本矩阵
                cost_matrix = self._compute_cost_matrix_progressive(
                    pred_box, pred_cls, pred_obj,
                    gt_boxes[b], gt_classes[b] if b < len(gt_classes) else None,
                    epoch, use_objectness, distance_threshold
                )

                # 为每个GT分配top-k预测
                matched_gt_mask = torch.zeros(n_gt, dtype=torch.bool, device=device)

                for gt_idx in range(n_gt):
                    costs = cost_matrix[:, gt_idx]

                    # 过滤距离过远的匹配
                    valid_mask = costs < distance_threshold * 10  # 转换为合理的成本阈值

                    if valid_mask.sum() > 0:
                        valid_costs = costs[valid_mask]
                        valid_indices = torch.where(valid_mask)[0]

                        k = min(k_factor, len(valid_indices))
                        if k > 0:
                            _, topk_idx_in_valid = torch.topk(valid_costs, k, largest=False)
                            topk_idx = valid_indices[topk_idx_in_valid]
                            assignments[b, s, topk_idx] = gt_idx
                            matched_gt_mask[gt_idx] = True

                # 记录未匹配的GT
                unmatched_indices = torch.where(~matched_gt_mask)[0]
                if len(unmatched_indices) > 0:
                    unmatched_boxes = gt_boxes[b][unmatched_indices]
                    unmatched_classes = gt_classes[b][unmatched_indices] if gt_classes else None
                    all_unmatched_gts.append({
                        'batch_idx': b,
                        'seq_idx': s,
                        'boxes': unmatched_boxes,
                        'classes': unmatched_classes
                    })

        return assignments, all_unmatched_gts if all_unmatched_gts else None

    def _compute_cost_matrix_progressive(self, pred_box, pred_cls, pred_obj,
                                         gt_box, gt_cls, epoch,
                                         use_objectness=True, distance_threshold=3.0):
        """渐进式成本矩阵计算"""
        N = pred_box.shape[0]
        M = gt_box.shape[0]
        device = pred_box.device

        if N == 0 or M == 0:
            return torch.full((N, M), float('inf'), device=device)

        # 中心距离成本（始终使用）
        pred_centers = pred_box[:, :3].unsqueeze(1)
        gt_centers = gt_box[:, :3].unsqueeze(0)
        center_dist = torch.norm(pred_centers - gt_centers, dim=2)

        # 初始化成本矩阵
        cost_matrix = center_dist.clone()

        # 分类成本（如果有GT类别）
        if gt_cls is not None and gt_cls.numel() > 0:
            gt_cls_onehot = F.one_hot(gt_cls.long(), self.num_classes).float()
            cls_cost = F.binary_cross_entropy_with_logits(
                pred_cls.unsqueeze(1).expand(-1, M, -1),
                gt_cls_onehot.unsqueeze(0).expand(N, -1, -1),
                reduction='none'
            ).sum(dim=2)

            # 渐进式调整分类成本权重
            cls_weight = min(1.0, epoch / self.warmup_epochs)
            cost_matrix = cost_matrix + cls_weight * cls_cost

        # 置信度成本（可选）
        if use_objectness:
            obj_cost = (1 - pred_obj).unsqueeze(1).expand(-1, M)
            obj_weight = min(0.5, epoch / (self.warmup_epochs * 2))
            cost_matrix = cost_matrix + obj_weight * obj_cost

        # 尺寸成本（后期添加）
        if epoch >= self.warmup_epochs:
            pred_sizes = pred_box[:, 3:6].unsqueeze(1)
            gt_sizes = gt_box[:, 3:6].unsqueeze(0)
            size_dist = torch.norm(torch.log(pred_sizes / (gt_sizes + 1e-6)), dim=2)
            cost_matrix = cost_matrix + 0.5 * size_dist

        return cost_matrix

    def _objectness_loss_progressive(self, pred, assignments, epoch):
        """渐进式objectness损失"""
        B, S, N = assignments.shape
        device = pred['pred_scores'].device

        targets = (assignments >= 0).float()
        pred_obj = pred['pred_scores']

        # 动态调整正负样本权重
        n_pos = targets.sum()
        n_neg = (targets == 0).sum()

        if epoch < self.warmup_epochs:
            # 早期：更高的正样本权重
            pos_weight = min(10.0, n_neg / max(n_pos, 1.0))
        else:
            # 后期：逐渐降低正样本权重
            base_weight = n_neg / max(n_pos, 1.0)
            decay = min(1.0, (epoch - self.warmup_epochs) / self.warmup_epochs)
            pos_weight = base_weight * (1 - 0.5 * decay)

        pos_weight = torch.tensor(pos_weight, device=device)

        # 使用BCE with pos_weight
        loss = F.binary_cross_entropy_with_logits(
            pred_obj, targets, pos_weight=pos_weight, reduction='mean'
        )

        return loss

    def _unmatched_gt_loss(self, pred, unmatched_gts, device):
        """为未匹配的GT添加额外损失，鼓励更多预测"""
        if not unmatched_gts:
            return torch.tensor(0.0, device=device)

        total_loss = torch.tensor(0.0, device=device)

        for unmatched in unmatched_gts:
            b, s = unmatched['batch_idx'], unmatched['seq_idx']
            gt_boxes = unmatched['boxes']

            pred_boxes = pred['pred_boxes'][b, s]
            pred_obj = pred['pred_scores'][b, s]

            for gt_box in gt_boxes:
                gt_center = gt_box[:3]
                pred_centers = pred_boxes[:, :3]
                distances = torch.norm(pred_centers - gt_center, dim=1)

                k = min(5, len(distances))
                if k > 0:
                    nearest_dists, nearest_idx = distances.topk(k, largest=False)
                    nearest_obj = pred_obj[nearest_idx]

                    weight = torch.exp(-nearest_dists / 2.0)
                    loss = weight * (1 - nearest_obj)
                    total_loss += loss.mean()

        return total_loss / max(len(unmatched_gts), 1)

    def _class_loss_focal(self, pred, gt_classes, assignments):
        """Focal Loss for classification"""
        device = pred['pred_classes'].device
        B, S, N, C = pred['pred_classes'].shape

        loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            if b >= len(gt_classes):
                continue
            for s in range(S):
                logits = pred['pred_classes'][b, s]

                # 创建目标
                targets = torch.zeros((N, C), device=device)
                pos_mask = assignments[b, s] >= 0

                if pos_mask.any() and len(gt_classes[b]) > 0:
                    gt_indices = assignments[b, s, pos_mask]
                    valid_indices = gt_indices < len(gt_classes[b])

                    if valid_indices.any():
                        gt_indices = gt_indices[valid_indices]
                        pos_indices = pos_mask.nonzero().squeeze(1)[valid_indices]

                        for pred_idx, gt_idx in zip(pos_indices, gt_indices):
                            targets[pred_idx, gt_classes[b][gt_idx]] = 1.0

                # Focal Loss计算
                p = torch.sigmoid(logits)
                ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
                p_t = p * targets + (1 - p) * (1 - targets)
                focal_weight = (1 - p_t) ** self.focal_gamma

                # 对正样本应用alpha权重
                alpha_t = targets * self.focal_alpha + (1 - targets) * (1 - self.focal_alpha)

                loss += (alpha_t * focal_weight * ce_loss).sum()
                count += pos_mask.sum()

        return loss / max(count, 1)

    # 保留其他损失函数...
    def _center_loss(self, pred, gt_boxes, assignments):
        device = pred['pred_boxes'].device
        B, S, N = assignments.shape

        loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            if b >= len(gt_boxes):
                continue
            for s in range(S):
                mask = assignments[b, s] >= 0
                if not mask.any():
                    continue

                pred_centers = pred['pred_boxes'][b, s, mask, :3]
                gt_indices = assignments[b, s, mask]

                valid_indices = gt_indices < len(gt_boxes[b])
                if not valid_indices.any():
                    continue

                gt_indices = gt_indices[valid_indices]
                pred_centers = pred_centers[valid_indices]
                gt_centers = gt_boxes[b][gt_indices, :3]

                loss += F.smooth_l1_loss(pred_centers, gt_centers, reduction='sum')
                count += valid_indices.sum()

        return loss / max(count, 1)

    def _size_loss(self, pred, gt_boxes, assignments):
        device = pred['pred_boxes'].device
        B, S, N = assignments.shape

        loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            if b >= len(gt_boxes):
                continue
            for s in range(S):
                mask = assignments[b, s] >= 0
                if not mask.any():
                    continue

                pred_sizes = pred['pred_boxes'][b, s, mask, 3:6]
                gt_indices = assignments[b, s, mask]

                valid_indices = gt_indices < len(gt_boxes[b])
                if not valid_indices.any():
                    continue

                gt_indices = gt_indices[valid_indices]
                pred_sizes = pred_sizes[valid_indices]
                gt_sizes = gt_boxes[b][gt_indices, 3:6]

                pred_log = torch.log(pred_sizes.clamp(min=0.01))
                gt_log = torch.log(gt_sizes.clamp(min=0.01))

                loss += F.smooth_l1_loss(pred_log, gt_log, reduction='sum')
                count += valid_indices.sum()

        return loss / max(count, 1)

    def _angle_loss(self, pred, gt_boxes, assignments):
        device = pred['pred_boxes'].device
        B, S, N = assignments.shape

        loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            if b >= len(gt_boxes):
                continue
            for s in range(S):
                mask = assignments[b, s] >= 0
                if not mask.any():
                    continue

                pred_angle = pred['pred_boxes'][b, s, mask, 6]
                gt_indices = assignments[b, s, mask]

                valid_indices = gt_indices < len(gt_boxes[b])
                if not valid_indices.any():
                    continue

                gt_indices = gt_indices[valid_indices]
                pred_angle = pred_angle[valid_indices]
                gt_angle = gt_boxes[b][gt_indices, 6]

                loss += F.smooth_l1_loss(torch.sin(pred_angle), torch.sin(gt_angle)) + \
                        F.smooth_l1_loss(torch.cos(pred_angle), torch.cos(gt_angle))
                count += valid_indices.sum()

        return loss / max(count, 1) * 0.5

    def _diou_loss(self, pred, gt_boxes, assignments):
        device = pred['pred_boxes'].device
        B, S, N = assignments.shape

        loss = torch.tensor(0.0, device=device)
        count = 0

        for b in range(B):
            if b >= len(gt_boxes):
                continue
            for s in range(S):
                mask = assignments[b, s] >= 0
                if not mask.any():
                    continue

                pred_box = pred['pred_boxes'][b, s, mask]
                gt_indices = assignments[b, s, mask]

                valid_indices = gt_indices < len(gt_boxes[b])
                if not valid_indices.any():
                    continue

                gt_indices = gt_indices[valid_indices]
                pred_box = pred_box[valid_indices]
                gt_box = gt_boxes[b][gt_indices]

                try:
                    from training.iou_3d import diou_loss_3d
                    loss += diou_loss_3d(pred_box, gt_box)
                    count += valid_indices.sum()
                except Exception as e:
                    # 如果3D IoU不可用，使用简化版本
                    center_dist = torch.norm(pred_box[:, :3] - gt_box[:, :3], dim=1)
                    size_dist = torch.norm(
                        torch.log(pred_box[:, 3:6] / (gt_box[:, 3:6] + 1e-6)),
                        dim=1
                    )
                    loss += (center_dist + size_dist).sum()
                    count += valid_indices.sum()

        return loss / max(count, 1)

    def _compute_dn_loss(self, pred: Dict, gt_boxes: List, gt_classes: List, dn_meta: Dict) -> torch.Tensor:
        """计算DN loss"""
        device = pred['pred_boxes'].device
        total_loss = torch.tensor(0.0, device=device)

        if 'dense_features' not in pred:
            return total_loss

        dense_boxes = pred['dense_features']['boxes']
        dense_classes = pred['dense_features']['classes']
        dense_obj = pred['dense_features']['objectness']

        B = dense_boxes.shape[0]
        dn_num_group = dn_meta.get('dn_num_group', 100)

        for b in range(min(B, len(dn_meta['dn_positive_idx']))):
            dn_idx = dn_meta['dn_positive_idx'][b]
            if not dn_idx or b >= len(gt_boxes):
                continue

            gt_box = gt_boxes[b]
            if len(gt_box) == 0:
                continue

            num_gt = len(gt_box)
            for i in range(min(num_gt, len(dn_idx) // dn_num_group)):
                start_idx = i * dn_num_group
                end_idx = min(start_idx + dn_num_group, len(dn_idx))

                for j in range(start_idx, end_idx):
                    if j < dense_boxes.shape[2]:
                        pred_box = dense_boxes[b, 0, j]
                        target_box = gt_box[i]

                        box_loss = F.l1_loss(pred_box[:3], target_box[:3])
                        box_loss += F.l1_loss(pred_box[3:6], target_box[3:6])
                        box_loss += F.l1_loss(torch.sin(pred_box[6]), torch.sin(target_box[6]))
                        box_loss += F.l1_loss(torch.cos(pred_box[6]), torch.cos(target_box[6]))

                        total_loss += box_loss / (dn_num_group * num_gt)

        return total_loss

    def _prepare_and_validate_gt(self, batch: Dict, expected_batch_size: int, device: torch.device):
        """准备并验证GT数据"""
        gt_boxes = batch.get('gt_boxes', [])
        gt_classes = batch.get('gt_classes', [])

        gt_boxes = self._normalize_gt_list(gt_boxes, expected_batch_size, device, 'boxes')
        gt_classes = self._normalize_gt_list(gt_classes, expected_batch_size, device, 'classes')

        return gt_boxes, gt_classes

    def _normalize_gt_list(self, gt_data, expected_size: int, device: torch.device, data_type: str):
        """标准化GT数据列表"""
        if gt_data is None:
            gt_data = []

        if not isinstance(gt_data, list):
            if torch.is_tensor(gt_data):
                if gt_data.ndim == 3 and gt_data.shape[0] == expected_size:
                    gt_data = [gt_data[i] for i in range(expected_size)]
                else:
                    gt_data = [gt_data for _ in range(expected_size)]
            else:
                gt_data = [gt_data] * expected_size

        current_len = len(gt_data)
        if current_len < expected_size:
            for _ in range(expected_size - current_len):
                if data_type == 'boxes':
                    gt_data.append(torch.zeros((0, 7), dtype=torch.float32, device=device))
                else:
                    gt_data.append(torch.zeros((0,), dtype=torch.long, device=device))
        elif current_len > expected_size:
            gt_data = gt_data[:expected_size]

        for i in range(len(gt_data)):
            item = gt_data[i]

            if not torch.is_tensor(item):
                if item is None:
                    if data_type == 'boxes':
                        item = torch.zeros((0, 7), dtype=torch.float32)
                    else:
                        item = torch.zeros((0,), dtype=torch.long)
                else:
                    if data_type == 'boxes':
                        item = torch.tensor(item, dtype=torch.float32)
                    else:
                        item = torch.tensor(item, dtype=torch.long)

            gt_data[i] = item.to(device)

        return gt_data


class MultitaskLoss(nn.Module):
    def __init__(self, detection=None, **kwargs):
        super().__init__()
        self.losses = nn.ModuleDict()

        if detection:
            weight = detection.pop('weight', 5.0)
            self.losses['detection'] = DetectionLoss(**detection)
            self.weight = weight

    def forward(self, pred: Dict, batch: Dict, epoch: int = 0) -> Dict[str, torch.Tensor]:
        losses = {}

        if 'detection' in self.losses and 'pred_boxes' in pred:
            det_losses = self.losses['detection'](pred, batch, epoch)
            for k, v in det_losses.items():
                losses[k] = v * self.weight

        losses['loss_objective'] = sum(v for k, v in losses.items() if 'metric' not in k)

        return losses