# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from typing import Union

ArrayLike = Union[np.ndarray, torch.Tensor]


def _is_numpy(x: ArrayLike) -> bool:
    return isinstance(x, np.ndarray)


def _is_torch(x: ArrayLike) -> bool:
    return isinstance(x, torch.Tensor)


def _ensure_torch(x: ArrayLike) -> torch.Tensor:
    """如果输入不是torch张量，则将其转换为torch张量。"""
    if _is_numpy(x):
        return torch.from_numpy(x)
    elif _is_torch(x):
        return x
    else:
        return torch.tensor(x)


def single_undistortion(params, tracks_normalized):
    """
    使用给定的畸变参数对归一化轨迹进行一次去畸变处理。

    参数:
        params (torch.Tensor or numpy.ndarray): 形状为BxN的畸变参数。
        tracks_normalized (torch.Tensor or numpy.ndarray): 形状为[batch_size, num_tracks, 2]的归一化轨迹张量。

    返回:
        torch.Tensor: 去畸变后的归一化轨迹张量。
    """
    params = _ensure_torch(params)
    tracks_normalized = _ensure_torch(tracks_normalized)

    u, v = tracks_normalized[..., 0].clone(), tracks_normalized[..., 1].clone()
    u_undist, v_undist = apply_distortion(params, u, v)
    return torch.stack([u_undist, v_undist], dim=-1)


def iterative_undistortion(params, tracks_normalized, max_iterations=100, max_step_norm=1e-10, rel_step_size=1e-6):
    """
    使用给定的畸变参数迭代地对归一化轨迹进行去畸变处理。

    参数:
        params (torch.Tensor or numpy.ndarray): 形状为BxN的畸变参数。
        tracks_normalized (torch.Tensor or numpy.ndarray): 形状为[batch_size, num_tracks, 2]的归一化轨迹张量。
        max_iterations (int): 去畸变过程的最大迭代次数。
        max_step_norm (float): 收敛的最大步长范数。
        rel_step_size (float): 数值微分的相对步长。

    返回:
        torch.Tensor: 去畸变后的归一化轨迹张量。
    """
    params = _ensure_torch(params)
    tracks_normalized = _ensure_torch(tracks_normalized)

    B, N, _ = tracks_normalized.shape
    u, v = tracks_normalized[..., 0].clone(), tracks_normalized[..., 1].clone()
    original_u, original_v = u.clone(), v.clone()

    eps = torch.finfo(u.dtype).eps
    for idx in range(max_iterations):
        u_undist, v_undist = apply_distortion(params, u, v)
        dx = original_u - u_undist
        dy = original_v - v_undist

        step_u = torch.clamp(torch.abs(u) * rel_step_size, min=eps)
        step_v = torch.clamp(torch.abs(v) * rel_step_size, min=eps)

        J_00 = (apply_distortion(params, u + step_u, v)[0] - apply_distortion(params, u - step_u, v)[0]) / (2 * step_u)
        J_01 = (apply_distortion(params, u, v + step_v)[0] - apply_distortion(params, u, v - step_v)[0]) / (2 * step_v)
        J_10 = (apply_distortion(params, u + step_u, v)[1] - apply_distortion(params, u - step_u, v)[1]) / (2 * step_u)
        J_11 = (apply_distortion(params, u, v + step_v)[1] - apply_distortion(params, u, v - step_v)[1]) / (2 * step_v)

        J = torch.stack([torch.stack([J_00 + 1, J_01], dim=-1), torch.stack([J_10, J_11 + 1], dim=-1)], dim=-2)

        delta = torch.linalg.solve(J, torch.stack([dx, dy], dim=-1))

        u += delta[..., 0]
        v += delta[..., 1]

        if torch.max((delta**2).sum(dim=-1)) < max_step_norm:
            break

    return torch.stack([u, v], dim=-1)


def apply_distortion(extra_params, u, v):
    """
    对给定的2D点应用径向或OpenCV畸变。

    参数:
        extra_params (torch.Tensor or numpy.ndarray): 形状为BxN的畸变参数，其中N可以是1、2或4。
        u (torch.Tensor or numpy.ndarray): 形状为Bxnum_tracks的归一化x坐标。
        v (torch.Tensor or numpy.ndarray): 形状为Bxnum_tracks的归一化y坐标。

    返回:
        points2D (torch.Tensor): 形状为BxNx2的畸变后2D点。
    """
    extra_params = _ensure_torch(extra_params)
    u = _ensure_torch(u)
    v = _ensure_torch(v)

    num_params = extra_params.shape[1]

    if num_params == 1:
        # 简单径向畸变
        k = extra_params[:, 0]
        u2 = u * u
        v2 = v * v
        r2 = u2 + v2
        radial = k[:, None] * r2
        du = u * radial
        dv = v * radial

    elif num_params == 2:
        # RadialCameraModel畸变
        k1, k2 = extra_params[:, 0], extra_params[:, 1]
        u2 = u * u
        v2 = v * v
        r2 = u2 + v2
        radial = k1[:, None] * r2 + k2[:, None] * r2 * r2
        du = u * radial
        dv = v * radial

    elif num_params == 4:
        # OpenCVCameraModel畸变
        k1, k2, p1, p2 = (extra_params[:, 0], extra_params[:, 1], extra_params[:, 2], extra_params[:, 3])
        u2 = u * u
        v2 = v * v
        uv = u * v
        r2 = u2 + v2
        radial = k1[:, None] * r2 + k2[:, None] * r2 * r2
        du = u * radial + 2 * p1[:, None] * uv + p2[:, None] * (r2 + 2 * u2)
        dv = v * radial + 2 * p2[:, None] * uv + p1[:, None] * (r2 + 2 * v2)
    else:
        raise ValueError("不支持的畸变参数数量")

    u = u.clone() + du
    v = v.clone() + dv

    return u, v


if __name__ == "__main__":
    import random
    import pycolmap

    max_diff = 0
    for i in range(1000):
        # 定义畸变参数（为简单起见，假设使用1个参数）
        B = random.randint(1, 500)
        track_num = random.randint(100, 1000)
        params = torch.rand((B, 1), dtype=torch.float32)  # 批大小1，4个参数
        tracks_normalized = torch.rand((B, track_num, 2), dtype=torch.float32)  # 批大小1，5个点

        # 对轨迹进行去畸变
        undistorted_tracks = iterative_undistortion(params, tracks_normalized)

        for b in range(B):
            pycolmap_intri = np.array([1, 0, 0, params[b].item()])
            pycam = pycolmap.Camera(model="SIMPLE_RADIAL", width=1, height=1, params=pycolmap_intri, camera_id=0)

            undistorted_tracks_pycolmap = pycam.cam_from_img(tracks_normalized[b].numpy())
            diff = (undistorted_tracks[b] - undistorted_tracks_pycolmap).abs().median()
            max_diff = max(max_diff, diff)
            print(f"差值: {diff}, 最大差值: {max_diff}")

    import pdb

    pdb.set_trace()