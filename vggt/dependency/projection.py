# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from .distortion import apply_distortion


def img_from_cam_np(
    intrinsics: np.ndarray, points_cam: np.ndarray, extra_params: np.ndarray | None = None, default: float = 0.0
) -> np.ndarray:
    """
    对相机空间点应用内参（和可选的径向畸变）。

    参数
    ----
    intrinsics  : (B,3,3) 相机矩阵K。
    points_cam  : (B,3,N) 齐次相机坐标 (x, y, z)ᵀ。
    extra_params: (B, N) 或 (B, k) 畸变参数 (k = 1,2,4) 或 None。
    default     : 用于np.nan替换的值。

    返回
    -------
    points2D : (B,N,2) 像素坐标。
    """
    # 1. 透视除法 ───────────────────────────────────────
    z = points_cam[:, 2:3, :]  # (B,1,N)
    points_cam_norm = points_cam / z  # (B,3,N)
    uv = points_cam_norm[:, :2, :]  # (B,2,N)

    # 2. 可选畸变 ──────────────────────────────────────
    if extra_params is not None:
        uu, vv = apply_distortion(extra_params, uv[:, 0], uv[:, 1])
        uv = np.stack([uu, vv], axis=1)  # (B,2,N)

    # 3. 齐次坐标然后K矩阵乘法 ─────────────────
    ones = np.ones_like(uv[:, :1, :])  # (B,1,N)
    points_cam_h = np.concatenate([uv, ones], axis=1)  # (B,3,N)

    # 批量矩阵乘法: K · [u v 1]ᵀ
    points2D_h = np.einsum("bij,bjk->bik", intrinsics, points_cam_h)  # (B,3,N)
    points2D = np.nan_to_num(points2D_h[:, :2, :], nan=default)  # (B,2,N)

    return points2D.transpose(0, 2, 1)  # (B,N,2)


def project_3D_points_np(
    points3D: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray | None = None,
    extra_params: np.ndarray | None = None,
    *,
    default: float = 0.0,
    only_points_cam: bool = False,
):
    """
    ``project_3D_points``的NumPy克隆版本。

    参数
    ----------
    points3D          : (N,3) 世界空间点。
    extrinsics        : (B,3,4) 每个B相机的[R|t]矩阵。
    intrinsics        : (B,3,3) K矩阵（如果只需要相机空间则可选）。
    extra_params      : (B,k) 或 (B,N) 畸变参数 (k ∈ {1,2,4}) 或 None。
    default           : 用于替换NaN的值。
    only_points_cam   : 如果为True，跳过投影并返回points_cam，points2D返回None。

    返回
    -------
    (points2D, points_cam) : 一个元组，其中points2D是(B,N,2)像素坐标，如果only_points_cam=True则为None，
                           points_cam是(B,3,N)相机空间坐标。
    """
    # ----- 0. 准备尺寸 -----------------------------------------------------
    N = points3D.shape[0]  # 点数
    B = extrinsics.shape[0]  # 相机数

    # ----- 1. 世界坐标 → 齐次坐标 -------------------------------------------
    w_h = np.ones((N, 1), dtype=points3D.dtype)
    points3D_h = np.concatenate([points3D, w_h], axis=1)  # (N,4)

    # 广播到每个相机（使用np.broadcast_to不会实际复制） ------
    points3D_h_B = np.broadcast_to(points3D_h, (B, N, 4))  # (B,N,4)

    # ----- 2. 应用外参（相机框架） ------------------------------
    # X_cam = E · X_hom
    # einsum:  E_(b i j)  ·  X_(b n j)  →  (b n i)
    points_cam = np.einsum("bij,bnj->bni", extrinsics, points3D_h_B)  # (B,N,3)
    points_cam = points_cam.transpose(0, 2, 1)  # (B,3,N)

    if only_points_cam:
        return None, points_cam

    # ----- 3. 内参 + 畸变 ---------------------------------------
    if intrinsics is None:
        raise ValueError("除非only_points_cam=True，否则必须提供`intrinsics`")

    points2D = img_from_cam_np(intrinsics, points_cam, extra_params=extra_params, default=default)

    return points2D, points_cam


def project_3D_points(points3D, extrinsics, intrinsics=None, extra_params=None, default=0, only_points_cam=False):
    """
    使用外参和内参将3D点转换为2D。
    参数:
        points3D (torch.Tensor): 形状为Px3的3D点。
        extrinsics (torch.Tensor): 形状为Bx3x4的外参。
        intrinsics (torch.Tensor): 形状为Bx3x3的内参。
        extra_params (torch.Tensor): 形状为BxN的额外参数，用于径向畸变。
        default (float): 用于替换NaN的默认值。
        only_points_cam (bool): 如果为True，跳过投影并返回points2D为None。

    返回:
        tuple: (points2D, points_cam) 其中points2D的形状为BxNx2，如果only_points_cam=True则为None，
               points_cam的形状为Bx3xN。
    """
    with torch.cuda.amp.autocast(dtype=torch.double):
        N = points3D.shape[0]  # 点的数量
        B = extrinsics.shape[0]  # 批大小，即相机数量
        points3D_homogeneous = torch.cat([points3D, torch.ones_like(points3D[..., 0:1])], dim=1)  # Nx4
        # 为批处理重塑
        points3D_homogeneous = points3D_homogeneous.unsqueeze(0).expand(B, -1, -1)  # BxNx4

        # 步骤1：应用外参
        # 将3D点转换到所有相机的相机坐标系
        points_cam = torch.bmm(extrinsics, points3D_homogeneous.transpose(-1, -2))

        if only_points_cam:
            return None, points_cam

        # 步骤2：应用内参和（可选）畸变
        points2D = img_from_cam(intrinsics, points_cam, extra_params, default)

        return points2D, points_cam


def img_from_cam(intrinsics, points_cam, extra_params=None, default=0.0):
    """
    对给定的3D点应用内参和可选畸变。

    参数:
        intrinsics (torch.Tensor): 形状为Bx3x3的相机内参。
        points_cam (torch.Tensor): 形状为Bx3xN的相机坐标系中的3D点。
        extra_params (torch.Tensor, optional): 形状为BxN的畸变参数，其中N可以是1、2或4。
        default (float, optional): 用于替换输出中NaN的默认值。

    返回:
        points2D (torch.Tensor): 形状为BxNx2的像素坐标中的2D点。
    """

    # 通过第三个坐标进行归一化（齐次除法）
    points_cam = points_cam / points_cam[:, 2:3, :]
    # 提取uv
    uv = points_cam[:, :2, :]

    # 如果提供了extra_params则应用畸变
    if extra_params is not None:
        uu, vv = apply_distortion(extra_params, uv[:, 0], uv[:, 1])
        uv = torch.stack([uu, vv], dim=1)

    # 为批量矩阵乘法准备points_cam
    points_cam_homo = torch.cat((uv, torch.ones_like(uv[:, :1, :])), dim=1)  # Bx3xN
    # 使用批量矩阵乘法应用内参
    points2D_homo = torch.bmm(intrinsics, points_cam_homo)  # Bx3xN

    # 提取x和y坐标
    points2D = points2D_homo[:, :2, :]  # Bx2xN

    # 用默认值替换NaN
    points2D = torch.nan_to_num(points2D, nan=default)

    return points2D.transpose(1, 2)  # BxNx2


if __name__ == "__main__":
    # 设置示例输入
    B, N = 24, 10240

    for _ in range(100):
        points3D = np.random.rand(N, 3).astype(np.float64)
        extrinsics = np.random.rand(B, 3, 4).astype(np.float64)
        intrinsics = np.random.rand(B, 3, 3).astype(np.float64)

        # 转换为torch张量
        points3D_torch = torch.tensor(points3D)
        extrinsics_torch = torch.tensor(extrinsics)
        intrinsics_torch = torch.tensor(intrinsics)

        # 运行NumPy实现
        points2D_np, points_cam_np = project_3D_points_np(points3D, extrinsics, intrinsics)

        # 运行torch实现
        points2D_torch, points_cam_torch = project_3D_points(points3D_torch, extrinsics_torch, intrinsics_torch)

        # 将torch输出转换为numpy
        points2D_torch_np = points2D_torch.detach().numpy()
        points_cam_torch_np = points_cam_torch.detach().numpy()

        # 计算差异
        diff = np.abs(points2D_np - points2D_torch_np)
        print("NumPy和PyTorch实现之间的差异：")
        print(diff)

        # 检查最大误差
        max_diff = np.max(diff)
        print(f"最大差异：{max_diff}")

        if np.allclose(points2D_np, points2D_torch_np, atol=1e-6):
            print("实现匹配度很高。")
        else:
            print("检测到显著差异。")

        if points_cam_np is not None:
            points_cam_diff = np.abs(points_cam_np - points_cam_torch_np)
            print("NumPy和PyTorch相机空间坐标之间的差异：")
            print(points_cam_diff)

            # 检查最大误差
            max_cam_diff = np.max(points_cam_diff)
            print(f"最大相机空间坐标差异：{max_cam_diff}")

            if np.allclose(points_cam_np, points_cam_torch_np, atol=1e-6):
                print("相机空间坐标匹配度很高。")
            else:
                print("在相机空间坐标中检测到显著差异。")