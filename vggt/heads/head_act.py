# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn.functional as F


def activate_pose(pred_pose_enc, trans_act="linear", quat_act="linear", fl_act="linear"):
    """
    使用指定的激活函数激活姿态参数。

    参数：
        pred_pose_enc: 包含编码姿态参数的张量[平移、四元数、焦距]
        trans_act: 平移分量的激活类型
        quat_act: 四元数分量的激活类型
        fl_act: 焦距分量的激活类型

    返回：
        激活后的姿态参数张量
    """
    T = pred_pose_enc[..., :3]
    quat = pred_pose_enc[..., 3:7]
    fl = pred_pose_enc[..., 7:]  # 或fov

    T = base_pose_act(T, trans_act)
    quat = base_pose_act(quat, quat_act)
    fl = base_pose_act(fl, fl_act)  # 或fov

    pred_pose_enc = torch.cat([T, quat, fl], dim=-1)

    return pred_pose_enc


def base_pose_act(pose_enc, act_type="linear"):
    """
    对姿态参数应用基本激活函数。

    参数：
        pose_enc: 包含编码姿态参数的张量
        act_type: 激活类型（"linear"、"inv_log"、"exp"、"relu"）

    返回：
        激活后的姿态参数
    """
    if act_type == "linear":
        return pose_enc
    elif act_type == "inv_log":
        return inverse_log_transform(pose_enc)
    elif act_type == "exp":
        return torch.exp(pose_enc)
    elif act_type == "relu":
        return F.relu(pose_enc)
    else:
        raise ValueError(f"未知的act_type：{act_type}")


def activate_head(out, activation="norm_exp", conf_activation="expp1"):
    """
    处理网络输出以提取3D点和置信度值。

    参数：
        out: 网络输出张量 (B, C, H, W)
        activation: 3D点的激活类型
        conf_activation: 置信度值的激活类型

    返回：
        (3D点张量，置信度张量)元组
    """
    # 将通道从最后一维移动到第4维 => (B, H, W, C)
    fmap = out.permute(0, 2, 3, 1)  # 期望B,H,W,C

    # 分割为xyz（前C-1个通道）和置信度（最后一个通道）
    xyz = fmap[:, :, :, :-1]
    conf = fmap[:, :, :, -1]

    if activation == "norm_exp":
        d = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        xyz_normed = xyz / d
        pts3d = xyz_normed * torch.expm1(d)
    elif activation == "norm":
        pts3d = xyz / xyz.norm(dim=-1, keepdim=True)
    elif activation == "exp":
        pts3d = torch.exp(xyz)
    elif activation == "relu":
        pts3d = F.relu(xyz)
    elif activation == "inv_log":
        pts3d = inverse_log_transform(xyz)
    elif activation == "xy_inv_log":
        xy, z = xyz.split([2, 1], dim=-1)
        z = inverse_log_transform(z)
        pts3d = torch.cat([xy * z, z], dim=-1)
    elif activation == "sigmoid":
        pts3d = torch.sigmoid(xyz)
    elif activation == "linear":
        pts3d = xyz
    else:
        raise ValueError(f"未知的激活：{activation}")

    if conf_activation == "expp1":
        conf_out = 1 + conf.exp()
    elif conf_activation == "expp0":
        conf_out = conf.exp()
    elif conf_activation == "sigmoid":
        conf_out = torch.sigmoid(conf)
    else:
        raise ValueError(f"未知的conf_activation：{conf_activation}")

    return pts3d, conf_out


def inverse_log_transform(y):
    """
    应用逆对数变换：sign(y) * (exp(|y|) - 1)

    参数：
        y: 输入张量

    返回：
        变换后的张量
    """
    return torch.sign(y) * (torch.expm1(torch.abs(y)))