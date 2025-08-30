import torch
from typing import Optional
from mmdet3d.structures import LiDARInstance3DBoxes
from mmcv.ops import box_iou_rotated, nms_rotated, nms


def diou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor, aligned: bool = False) -> torch.Tensor:
    """计算3D DIoU"""
    assert boxes1.shape[-1] == 7 and boxes2.shape[-1] == 7

    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        size = (boxes1.shape[0],) if aligned else (boxes1.shape[0], boxes2.shape[0])
        return torch.zeros(size, device=boxes1.device)

    # 计算标准IoU
    boxes1_lidar = LiDARInstance3DBoxes(boxes1, origin=(0.5, 0.5, 0.5))
    boxes2_lidar = LiDARInstance3DBoxes(boxes2, origin=(0.5, 0.5, 0.5))
    iou = boxes1_lidar.overlaps(boxes2_lidar, mode='iou')

    # 计算中心距离
    if aligned:
        centers1 = boxes1[:, :3]
        centers2 = boxes2[:, :3]
        center_dist = torch.norm(centers1 - centers2, dim=1)

        # 计算包围盒对角线
        corners1 = boxes1_lidar.corners  # [N, 8, 3]
        corners2 = boxes2_lidar.corners

        all_corners = torch.cat([corners1, corners2], dim=1)  # [N, 16, 3]
        min_corner = all_corners.min(dim=1)[0]  # [N, 3]
        max_corner = all_corners.max(dim=1)[0]  # [N, 3]
        diagonal = torch.norm(max_corner - min_corner, dim=1)

        diou = torch.diagonal(iou) - (center_dist / diagonal.clamp(min=1e-6)) ** 2
    else:
        centers1 = boxes1[:, :3].unsqueeze(1)  # [N, 1, 3]
        centers2 = boxes2[:, :3].unsqueeze(0)  # [1, M, 3]
        center_dist = torch.norm(centers1 - centers2, dim=2)  # [N, M]

        # 简化：使用最大尺寸作为对角线近似
        max_size1 = boxes1[:, 3:6].max(dim=1)[0].unsqueeze(1)  # [N, 1]
        max_size2 = boxes2[:, 3:6].max(dim=1)[0].unsqueeze(0)  # [1, M]
        diagonal = (max_size1 + max_size2) * 2  # 近似包围盒对角线

        diou = iou - (center_dist / diagonal.clamp(min=1e-6)) ** 2

    return diou.clamp(min=0, max=1)


def giou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor, aligned: bool = True) -> torch.Tensor:
    """计算3D GIoU"""
    boxes1_lidar = LiDARInstance3DBoxes(boxes1, origin=(0.5, 0.5, 0.5))
    boxes2_lidar = LiDARInstance3DBoxes(boxes2, origin=(0.5, 0.5, 0.5))

    # 使用mmdet3d的GIoU实现
    giou = boxes1_lidar.overlaps(boxes2_lidar, mode='giou')

    return torch.diagonal(giou) if aligned else giou


def diou_loss_3d(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """3D DIoU损失"""
    diou = diou_3d(pred, target, aligned=True)
    return (1 - diou).mean()


def iou_3d(boxes1: torch.Tensor, boxes2: torch.Tensor, aligned: bool = False) -> torch.Tensor:
    """标准3D IoU"""
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        size = (boxes1.shape[0],) if aligned else (boxes1.shape[0], boxes2.shape[0])
        return torch.zeros(size, device=boxes1.device)

    boxes1_lidar = LiDARInstance3DBoxes(boxes1, origin=(0.5, 0.5, 0.5))
    boxes2_lidar = LiDARInstance3DBoxes(boxes2, origin=(0.5, 0.5, 0.5))
    iou = boxes1_lidar.overlaps(boxes2_lidar, mode='iou')

    return torch.diagonal(iou) if aligned else iou


def iou_loss_3d(pred: torch.Tensor, target: torch.Tensor, loss_type: str = 'iou') -> torch.Tensor:
    """Calculate 3D IoU loss."""
    iou = iou_3d(pred, target, aligned=True)

    if loss_type == 'giou':
        try:
            boxes1 = LiDARInstance3DBoxes(pred, origin=(0.5, 0.5, 0.5))
            boxes2 = LiDARInstance3DBoxes(target, origin=(0.5, 0.5, 0.5))
            giou_matrix = LiDARInstance3DBoxes.overlaps(boxes1, boxes2, mode='giou')
            iou = torch.diagonal(giou_matrix)
        except:
            pass

    return (1 - iou).mean()


def iou_bev(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Calculate BEV IoU."""

    def to_bev(boxes):
        return torch.stack([
            boxes[:, 0], boxes[:, 1],  # x, y
            boxes[:, 3], boxes[:, 4],  # width, height
            boxes[:, 6]  # yaw
        ], dim=1)

    return box_iou_rotated(to_bev(boxes1), to_bev(boxes2), mode='iou', aligned=False)


def nms_bev(boxes: torch.Tensor, scores: torch.Tensor,
            thresh: float = 0.5, max_num: Optional[int] = None) -> torch.Tensor:
    """BEV NMS."""
    order = scores.argsort(descending=True)
    if max_num:
        order = order[:max_num]

    boxes_sorted = boxes[order]
    scores_sorted = scores[order]

    boxes_bev = torch.stack([
        boxes_sorted[:, 0], boxes_sorted[:, 1],
        boxes_sorted[:, 3], boxes_sorted[:, 4],
        boxes_sorted[:, 6]
    ], dim=1)

    keep = nms_rotated(boxes_bev, scores_sorted, thresh)[1]
    return order[keep[:max_num] if max_num else keep]