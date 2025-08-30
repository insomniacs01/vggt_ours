# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# 修改自PyTorch3D，https://github.com/facebookresearch/pytorch3d

import torch
import numpy as np
import torch.nn.functional as F


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    四元数顺序：XYZW或称ijkr，标量在最后

    将给定为四元数的旋转转换为旋转矩阵。
    参数:
        quaternions: 实部在最后的四元数，
            张量形状为 (..., 4)。

    返回:
        旋转矩阵，张量形状为 (..., 3, 3)。
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` 不支持操作数类型 `float` 和 `Tensor`。
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    将给定为旋转矩阵的旋转转换为四元数。

    参数:
        matrix: 旋转矩阵，张量形状为 (..., 3, 3)。

    返回:
        实部在最后的四元数，张量形状为 (..., 4)。
        四元数顺序：XYZW或称ijkr，标量在最后
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"无效的旋转矩阵形状 {matrix.shape}。")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack(
            [1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22, 1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1
        )
    )

    # 我们生成由r、i、j、k中的每一个乘以的所需四元数
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` 不支持操作数类型 `Tensor` 和
            #  `int`。
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` 不支持操作数类型 `Tensor` 和
            #  `int`。
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` 不支持操作数类型 `Tensor` 和
            #  `int`。
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` 不支持操作数类型 `Tensor` 和
            #  `int`。
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # 我们在这里下限为0.1，但确切的水平并不重要；如果q_abs很小，
    # 候选者不会被选中。
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # 如果不是数值问题，quat_candidates[i]应该相同（最多相差一个符号），
    # 对所有i；我们选择条件最好的（具有最大分母）
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))

    # 从rijk转换为ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    返回 torch.sqrt(torch.max(0, x))
    但在x为0时具有零子梯度。
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    将单位四元数转换为标准形式：实部为非负的形式。

    参数:
        quaternions: 实部在最后的四元数，
            张量形状为 (..., 4)。

    返回:
        标准化的四元数，张量形状为 (..., 4)。
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)