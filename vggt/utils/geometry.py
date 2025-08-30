# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from vggt.dependency.distortion import apply_distortion, iterative_undistortion, single_undistortion


def unproject_depth_map_to_point_map(
        depth_map: np.ndarray, extrinsics_cam: np.ndarray, intrinsics_cam: np.ndarray
) -> np.ndarray:
    """
    将批量深度图反投影到3D世界坐标。

    参数:
        depth_map (np.ndarray): 深度图批量，形状为 (S, H, W, 1) 或 (S, H, W)
        extrinsics_cam (np.ndarray): 相机外参矩阵批量，形状为 (S, 3, 4)
        intrinsics_cam (np.ndarray): 相机内参矩阵批量，形状为 (S, 3, 3)

    返回:
        np.ndarray: 3D世界坐标批量，形状为 (S, H, W, 3)
    """
    if isinstance(depth_map, torch.Tensor):
        depth_map = depth_map.cpu().numpy()
    if isinstance(extrinsics_cam, torch.Tensor):
        extrinsics_cam = extrinsics_cam.cpu().numpy()
    if isinstance(intrinsics_cam, torch.Tensor):
        intrinsics_cam = intrinsics_cam.cpu().numpy()

    world_points_list = []
    for frame_idx in range(depth_map.shape[0]):
        cur_world_points, _, _ = depth_to_world_coords_points(
            depth_map[frame_idx].squeeze(-1), extrinsics_cam[frame_idx], intrinsics_cam[frame_idx]
        )
        world_points_list.append(cur_world_points)
    world_points_array = np.stack(world_points_list, axis=0)

    return world_points_array


def depth_to_world_coords_points(
        depth_map: np.ndarray,
        extrinsic: np.ndarray,
        intrinsic: np.ndarray,
        eps=1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将深度图转换为世界坐标。

    参数:
        depth_map (np.ndarray): 深度图，形状为 (H, W)。
        intrinsic (np.ndarray): 相机内参矩阵，形状为 (3, 3)。
        extrinsic (np.ndarray): 相机外参矩阵，形状为 (3, 4)。OpenCV相机坐标约定，从世界到相机。

    返回:
        tuple[np.ndarray, np.ndarray]: 世界坐标 (H, W, 3) 和有效深度掩码 (H, W)。
    """
    if depth_map is None:
        return None, None, None

    # 有效深度掩码
    point_mask = depth_map > eps

    # 将深度图转换为相机坐标
    cam_coords_points = depth_to_cam_coords_points(depth_map, intrinsic)

    # 乘以外参矩阵的逆以转换到世界坐标
    # extrinsic_inv 是 4x4（注意 closed_form_inverse_OpenCV 是批处理的，输出是 (N, 4, 4)）
    cam_to_world_extrinsic = closed_form_inverse_se3(extrinsic[None])[0]

    R_cam_to_world = cam_to_world_extrinsic[:3, :3]
    t_cam_to_world = cam_to_world_extrinsic[:3, 3]

    # 对相机坐标应用旋转和平移
    world_coords_points = np.dot(cam_coords_points, R_cam_to_world.T) + t_cam_to_world  # HxWx3, 3x3 -> HxWx3
    # world_coords_points = np.einsum("ij,hwj->hwi", R_cam_to_world, cam_coords_points) + t_cam_to_world

    return world_coords_points, cam_coords_points, point_mask


def depth_to_cam_coords_points(depth_map: np.ndarray, intrinsic: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将深度图转换为相机坐标。

    参数:
        depth_map (np.ndarray): 深度图，形状为 (H, W)。
        intrinsic (np.ndarray): 相机内参矩阵，形状为 (3, 3)。

    返回:
        tuple[np.ndarray, np.ndarray]: 相机坐标 (H, W, 3)
    """
    H, W = depth_map.shape
    assert intrinsic.shape == (3, 3), "内参矩阵必须是 3x3"
    assert intrinsic[0, 1] == 0 and intrinsic[1, 0] == 0, "内参矩阵必须没有歪斜"

    # 内参参数
    fu, fv = intrinsic[0, 0], intrinsic[1, 1]
    cu, cv = intrinsic[0, 2], intrinsic[1, 2]

    # 生成像素坐标网格
    u, v = np.meshgrid(np.arange(W), np.arange(H))

    # 反投影到相机坐标
    x_cam = (u - cu) * depth_map / fu
    y_cam = (v - cv) * depth_map / fv
    z_cam = depth_map

    # 堆叠形成相机坐标
    cam_coords = np.stack((x_cam, y_cam, z_cam), axis=-1).astype(np.float32)

    return cam_coords


def closed_form_inverse_se3(se3, R=None, T=None):
    """
    计算批量中每个4x4（或3x4）SE3矩阵的逆。

    如果提供了 `R` 和 `T`，它们必须对应于 `se3` 的旋转和平移
    分量。否则，它们将从 `se3` 中提取。

    参数:
        se3: Nx4x4 或 Nx3x4 的SE3矩阵数组或张量。
        R (可选): Nx3x3 的旋转矩阵数组或张量。
        T (可选): Nx3x1 的平移向量数组或张量。

    返回:
        与 `se3` 具有相同类型和设备的逆SE3矩阵。

    形状:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # 检查se3是numpy数组还是torch张量
    is_numpy = isinstance(se3, np.ndarray)

    # 验证形状
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3必须是形状(N,4,4)，得到 {se3.shape}。")

    # 如果未提供，则提取R和T
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # 转置R
    if is_numpy:
        # 计算NumPy的旋转转置
        R_transposed = np.transpose(R, (0, 2, 1))
        # NumPy的 -R^T t
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix


# TODO: 这段代码可以进一步清理


def project_world_points_to_camera_points_batch(world_points, cam_extrinsics):
    """
    使用外参和内参将3D点转换为2D。
    参数:
        world_points (torch.Tensor): 3D点，形状为 BxSxHxWx3。
        cam_extrinsics (torch.Tensor): 外参参数，形状为 BxSx3x4。
    返回:
    """
    # TODO: 将此合并到 project_world_points_to_cam 中

    # device = world_points.device
    # with torch.autocast(device_type=device.type, enabled=False):
    ones = torch.ones_like(world_points[..., :1])  # 形状: (B, S, H, W, 1)
    world_points_h = torch.cat([world_points, ones], dim=-1)  # 形状: (B, S, H, W, 4)

    # 外参: (B, S, 3, 4) -> (B, S, 1, 1, 3, 4)
    extrinsics_exp = cam_extrinsics.unsqueeze(2).unsqueeze(3)

    # world_points_h: (B, S, H, W, 4) -> (B, S, H, W, 4, 1)
    world_points_h_exp = world_points_h.unsqueeze(-1)

    # 现在执行矩阵乘法
    # (B, S, 1, 1, 3, 4) @ (B, S, H, W, 4, 1) 广播到 (B, S, H, W, 3, 1)
    camera_points = torch.matmul(extrinsics_exp, world_points_h_exp).squeeze(-1)

    return camera_points


def project_world_points_to_cam(
        world_points,
        cam_extrinsics,
        cam_intrinsics=None,
        distortion_params=None,
        default=0,
        only_points_cam=False,
):
    """
    使用外参和内参将3D点转换为2D。
    参数:
        world_points (torch.Tensor): 3D点，形状为 Px3。
        cam_extrinsics (torch.Tensor): 外参参数，形状为 Bx3x4。
        cam_intrinsics (torch.Tensor): 内参参数，形状为 Bx3x3。
        distortion_params (torch.Tensor): 额外参数，形状为 BxN，用于径向畸变。
    返回:
        torch.Tensor: 转换后的2D点，形状为 BxNx2。
    """
    device = world_points.device
    # with torch.autocast(device_type=device.type, dtype=torch.double):
    with torch.autocast(device_type=device.type, enabled=False):
        N = world_points.shape[0]  # 点的数量
        B = cam_extrinsics.shape[0]  # 批量大小，即相机数量
        world_points_homogeneous = torch.cat(
            [world_points, torch.ones_like(world_points[..., 0:1])], dim=1
        )  # Nx4
        # 为批处理重塑
        world_points_homogeneous = world_points_homogeneous.unsqueeze(0).expand(
            B, -1, -1
        )  # BxNx4

        # 步骤1：应用外参参数
        # 将3D点转换到所有相机的相机坐标系
        cam_points = torch.bmm(
            cam_extrinsics, world_points_homogeneous.transpose(-1, -2)
        )

        if only_points_cam:
            return None, cam_points

        # 步骤2：应用内参参数和（可选）畸变
        image_points = img_from_cam(cam_intrinsics, cam_points, distortion_params, default=default)

        return image_points, cam_points


def img_from_cam(cam_intrinsics, cam_points, distortion_params=None, default=0.0):
    """
    对给定的3D点应用内参参数和可选畸变。

    参数:
        cam_intrinsics (torch.Tensor): 相机内参参数，形状为 Bx3x3。
        cam_points (torch.Tensor): 相机坐标中的3D点，形状为 Bx3xN。
        distortion_params (torch.Tensor, 可选): 畸变参数，形状为 BxN，其中N可以是1、2或4。
        default (float, 可选): 用于替换输出中NaN的默认值。

    返回:
        pixel_coords (torch.Tensor): 像素坐标中的2D点，形状为 BxNx2。
    """

    # 归一化设备坐标（NDC）
    cam_points = cam_points / cam_points[:, 2:3, :]
    ndc_xy = cam_points[:, :2, :]

    # 如果提供了畸变参数，则应用畸变
    if distortion_params is not None:
        x_distorted, y_distorted = apply_distortion(distortion_params, ndc_xy[:, 0], ndc_xy[:, 1])
        distorted_xy = torch.stack([x_distorted, y_distorted], dim=1)
    else:
        distorted_xy = ndc_xy

    # 为批量矩阵乘法准备cam_points
    cam_coords_homo = torch.cat(
        (distorted_xy, torch.ones_like(distorted_xy[:, :1, :])), dim=1
    )  # Bx3xN
    # 使用批量矩阵乘法应用内参参数
    pixel_coords = torch.bmm(cam_intrinsics, cam_coords_homo)  # Bx3xN

    # 提取x和y坐标
    pixel_coords = pixel_coords[:, :2, :]  # Bx2xN

    # 用默认值替换NaN
    pixel_coords = torch.nan_to_num(pixel_coords, nan=default)

    return pixel_coords.transpose(1, 2)  # BxNx2


def cam_from_img(pred_tracks, intrinsics, extra_params=None):
    """
    基于相机内参归一化预测的轨迹。
    参数:
    intrinsics (torch.Tensor): 相机内参张量，形状为 [batch_size, 3, 3]。
    pred_tracks (torch.Tensor): 预测的轨迹张量，形状为 [batch_size, num_tracks, 2]。
    extra_params (torch.Tensor, 可选): 畸变参数，形状为 BxN，其中N可以是1、2或4。
    返回:
    torch.Tensor: 归一化的轨迹张量。
    """

    # 我们不想在这里执行 intrinsics_inv = torch.inverse(intrinsics)
    # 否则我们可以使用类似
    #     tracks_normalized_homo = torch.bmm(pred_tracks_homo, intrinsics_inv.transpose(1, 2))

    principal_point = intrinsics[:, [0, 1], [2, 2]].unsqueeze(-2)
    focal_length = intrinsics[:, [0, 1], [0, 1]].unsqueeze(-2)
    tracks_normalized = (pred_tracks - principal_point) / focal_length

    if extra_params is not None:
        # 应用迭代去畸变
        try:
            tracks_normalized = iterative_undistortion(
                extra_params, tracks_normalized
            )
        except:
            tracks_normalized = single_undistortion(
                extra_params, tracks_normalized
            )

    return tracks_normalized