# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .rotation import quat_to_mat, mat_to_quat


def extri_intri_to_pose_encoding(
    extrinsics, intrinsics, image_size_hw=None, pose_encoding_type="absT_quaR_FoV"  # 例如，(256, 512)
):
    """将相机外参和内参转换为紧凑的位姿编码。

    此函数将相机参数转换为统一的位姿编码格式，
    可用于各种下游任务，如位姿预测或表示。

    参数:
        extrinsics (torch.Tensor): 相机外参参数，形状为 BxSx3x4，
            其中B是批量大小，S是序列长度。
            在OpenCV坐标系中（x-右，y-下，z-前），表示从世界到相机的变换。
            格式为[R|t]，其中R是3x3旋转矩阵，t是3x1平移向量。
        intrinsics (torch.Tensor): 相机内参参数，形状为 BxSx3x3。
            以像素为单位定义，格式为：
            [[fx, 0, cx],
             [0, fy, cy],
             [0,  0,  1]]
            其中fx、fy是焦距，(cx, cy)是主点
        image_size_hw (tuple): 图像的(高度, 宽度)元组，以像素为单位。
            计算视场值所需。例如：(256, 512)。
        pose_encoding_type (str): 要使用的位姿编码类型。目前仅
            支持"absT_quaR_FoV"（绝对平移、四元数旋转、视场）。

    返回:
        torch.Tensor: 编码的相机位姿参数，形状为 BxSx9。
            对于"absT_quaR_FoV"类型，9个维度为：
            - [:3] = 绝对平移向量T (3D)
            - [3:7] = 四元数旋转quat (4D)
            - [7:] = 视场 (2D)
    """

    # 外参: BxSx3x4
    # 内参: BxSx3x3

    if pose_encoding_type == "absT_quaR_FoV":
        R = extrinsics[:, :, :3, :3]  # BxSx3x3
        T = extrinsics[:, :, :3, 3]  # BxSx3

        quat = mat_to_quat(R)
        # 注意这里h和w的顺序
        H, W = image_size_hw
        fov_h = 2 * torch.atan((H / 2) / intrinsics[..., 1, 1])
        fov_w = 2 * torch.atan((W / 2) / intrinsics[..., 0, 0])
        pose_encoding = torch.cat([T, quat, fov_h[..., None], fov_w[..., None]], dim=-1).float()
    else:
        raise NotImplementedError

    return pose_encoding


def pose_encoding_to_extri_intri(
    pose_encoding, image_size_hw=None, pose_encoding_type="absT_quaR_FoV", build_intrinsics=True  # 例如，(256, 512)
):
    """将位姿编码转换回相机外参和内参。

    此函数执行extri_intri_to_pose_encoding的逆操作，
    从紧凑编码重建完整的相机参数。

    参数:
        pose_encoding (torch.Tensor): 编码的相机位姿参数，形状为 BxSx9，
            其中B是批量大小，S是序列长度。
            对于"absT_quaR_FoV"类型，9个维度为：
            - [:3] = 绝对平移向量T (3D)
            - [3:7] = 四元数旋转quat (4D)
            - [7:] = 视场 (2D)
        image_size_hw (tuple): 图像的(高度, 宽度)元组，以像素为单位。
            从视场值重建内参所需。
            例如：(256, 512)。
        pose_encoding_type (str): 使用的位姿编码类型。目前仅
            支持"absT_quaR_FoV"（绝对平移、四元数旋转、视场）。
        build_intrinsics (bool): 是否重建内参矩阵。
            如果为False，则仅返回外参，内参将为None。

    返回:
        tuple: (extrinsics, intrinsics)
            - extrinsics (torch.Tensor): 相机外参参数，形状为 BxSx3x4。
              在OpenCV坐标系中（x-右，y-下，z-前），表示从世界到相机
              的变换。格式为[R|t]，其中R是3x3旋转矩阵，t是
              3x1平移向量。
            - intrinsics (torch.Tensor 或 None): 相机内参参数，形状为 BxSx3x3，
              如果build_intrinsics为False则为None。以像素为单位定义，格式为：
              [[fx, 0, cx],
               [0, fy, cy],
               [0,  0,  1]]
              其中fx、fy是焦距，(cx, cy)是主点，
              假设在图像中心(W/2, H/2)。
    """

    intrinsics = None

    if pose_encoding_type == "absT_quaR_FoV":
        T = pose_encoding[..., :3]
        quat = pose_encoding[..., 3:7]
        fov_h = pose_encoding[..., 7]
        fov_w = pose_encoding[..., 8]

        R = quat_to_mat(quat)
        extrinsics = torch.cat([R, T[..., None]], dim=-1)

        if build_intrinsics:
            H, W = image_size_hw
            fy = (H / 2.0) / torch.tan(fov_h / 2.0)
            fx = (W / 2.0) / torch.tan(fov_w / 2.0)
            intrinsics = torch.zeros(pose_encoding.shape[:2] + (3, 3), device=pose_encoding.device)
            intrinsics[..., 0, 0] = fx
            intrinsics[..., 1, 1] = fy
            intrinsics[..., 0, 2] = W / 2
            intrinsics[..., 1, 2] = H / 2
            intrinsics[..., 2, 2] = 1.0  # 将齐次坐标设置为1
    else:
        raise NotImplementedError

    return extrinsics, intrinsics