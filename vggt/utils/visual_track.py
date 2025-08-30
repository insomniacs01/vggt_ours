# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import cv2
import torch
import numpy as np
import os


def color_from_xy(x, y, W, H, cmap_name="hsv"):
    """
    映射 (x, y) -> 颜色 (R, G, B)。
    1) 将x,y归一化到[0,1]。
    2) 将它们组合成[0,1]中的单个标量c。
    3) 使用matplotlib的颜色映射将c转换为(R,G,B)。

    您可以自定义步骤2，例如，c = (x + y)/2，或(x, y)的某个函数。
    """
    import matplotlib.cm
    import matplotlib.colors

    x_norm = x / max(W - 1, 1)
    y_norm = y / max(H - 1, 1)
    # 简单组合：
    c = (x_norm + y_norm) / 2.0

    cmap = matplotlib.cm.get_cmap(cmap_name)
    # cmap(c) -> (r,g,b,a) 在 [0,1]
    rgba = cmap(c)
    r, g, b = rgba[0], rgba[1], rgba[2]
    return (r, g, b)  # 在[0,1]，RGB顺序


def get_track_colors_by_position(tracks_b, vis_mask_b=None, image_width=None, image_height=None, cmap_name="hsv"):
    """
    给定一个样本(b)中的所有轨迹，计算(N,3)的RGB颜色值数组，
    范围在[0,255]。颜色由每个轨迹在第一个
    可见帧中的(x,y)位置决定。

    参数:
        tracks_b: 形状为(S, N, 2)的张量。每帧中每个轨迹的(x,y)。
        vis_mask_b: (S, N)布尔掩码；如果为None，假设全部可见。
        image_width, image_height: 用于归一化(x, y)。
        cmap_name: matplotlib的颜色映射（例如，'hsv'、'rainbow'、'jet'）。

    返回:
        track_colors: 形状为(N, 3)的np.ndarray，每行是[0,255]中的(R,G,B)。
    """
    S, N, _ = tracks_b.shape
    track_colors = np.zeros((N, 3), dtype=np.uint8)

    if vis_mask_b is None:
        # 将全部视为可见
        vis_mask_b = torch.ones(S, N, dtype=torch.bool, device=tracks_b.device)

    for i in range(N):
        # 查找轨迹i的第一个可见帧
        visible_frames = torch.where(vis_mask_b[:, i])[0]
        if len(visible_frames) == 0:
            # 轨迹从未可见；只分配黑色或其他颜色
            track_colors[i] = (0, 0, 0)
            continue

        first_s = int(visible_frames[0].item())
        # 使用该帧的(x,y)
        x, y = tracks_b[first_s, i].tolist()

        # 映射(x,y) -> (R,G,B)在[0,1]
        r, g, b = color_from_xy(x, y, W=image_width, H=image_height, cmap_name=cmap_name)
        # 缩放到[0,255]
        r, g, b = int(r * 255), int(g * 255), int(b * 255)
        track_colors[i] = (r, g, b)

    return track_colors


def visualize_tracks_on_images(
    images,
    tracks,
    track_vis_mask=None,
    out_dir="track_visuals_concat_by_xy",
    image_format="CHW",  # "CHW" 或 "HWC"
    normalize_mode="[0,1]",
    cmap_name="hsv",  # 例如 "hsv", "rainbow", "jet"
    frames_per_row=4,  # 网格布局的新参数
    save_grid=True,  # 控制是否保存网格图像的标志
):
    """
    以网格布局可视化帧，每行指定数量的帧。
    每个轨迹的颜色由其在第一个可见帧中的(x,y)位置决定
    （如果始终可见，则为第0帧）。
    最后在保存之前将BGR结果转换为RGB。
    还将每个单独的帧保存为单独的PNG文件。

    参数:
        images: torch.Tensor (S, 3, H, W) 如果CHW或 (S, H, W, 3) 如果HWC。
        tracks: torch.Tensor (S, N, 2)，最后一维 = (x, y)。
        track_vis_mask: torch.Tensor (S, N) 或 None。
        out_dir: 保存可视化的文件夹。
        image_format: "CHW" 或 "HWC"。
        normalize_mode: "[0,1]"、"[-1,1]" 或 None 用于直接原始 -> 0..255
        cmap_name: color_from_xy的matplotlib颜色映射名称。
        frames_per_row: 网格中每行显示的帧数。
        save_grid: 是否将所有帧保存在一个网格图像中。

    返回:
        None（在out_dir中保存图像）。
    """

    if len(tracks.shape) == 4:
        tracks = tracks.squeeze(0)
        images = images.squeeze(0)
        if track_vis_mask is not None:
            track_vis_mask = track_vis_mask.squeeze(0)

    import matplotlib

    matplotlib.use("Agg")  # 用于非交互式（可选）

    os.makedirs(out_dir, exist_ok=True)

    S = images.shape[0]
    _, N, _ = tracks.shape  # (S, N, 2)

    # 移动到CPU
    images = images.cpu().clone()
    tracks = tracks.cpu().clone()
    if track_vis_mask is not None:
        track_vis_mask = track_vis_mask.cpu().clone()

    # 从图像形状推断H、W
    if image_format == "CHW":
        # 例如 images[s].shape = (3, H, W)
        H, W = images.shape[2], images.shape[3]
    else:
        # 例如 images[s].shape = (H, W, 3)
        H, W = images.shape[1], images.shape[2]

    # 基于第一个可见位置预计算每个轨迹i的颜色
    track_colors_rgb = get_track_colors_by_position(
        tracks,  # 形状 (S, N, 2)
        vis_mask_b=track_vis_mask if track_vis_mask is not None else None,
        image_width=W,
        image_height=H,
        cmap_name=cmap_name,
    )

    # 我们将在列表中累积每帧的绘制图像
    frame_images = []

    for s in range(S):
        # 形状 => (3, H, W) 或 (H, W, 3)
        img = images[s]

        # 转换为 (H, W, 3)
        if image_format == "CHW":
            img = img.permute(1, 2, 0)  # (H, W, 3)
        # 否则 "HWC"，不做任何操作

        img = img.numpy().astype(np.float32)

        # 如果需要，缩放到[0,255]
        if normalize_mode == "[0,1]":
            img = np.clip(img, 0, 1) * 255.0
        elif normalize_mode == "[-1,1]":
            img = (img + 1.0) * 0.5 * 255.0
            img = np.clip(img, 0, 255.0)
        # 否则没有归一化

        # 转换为uint8
        img = img.astype(np.uint8)

        # 为了在OpenCV中绘制，转换为BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # 绘制每个可见轨迹
        cur_tracks = tracks[s]  # 形状 (N, 2)
        if track_vis_mask is not None:
            valid_indices = torch.where(track_vis_mask[s])[0]
        else:
            valid_indices = range(N)

        cur_tracks_np = cur_tracks.numpy()
        for i in valid_indices:
            x, y = cur_tracks_np[i]
            pt = (int(round(x)), int(round(y)))

            # track_colors_rgb[i] 是 (R,G,B)。对于OpenCV圆，我们需要BGR
            R, G, B = track_colors_rgb[i]
            color_bgr = (int(B), int(G), int(R))
            cv2.circle(img_bgr, pt, radius=3, color=color_bgr, thickness=-1)

        # 转换回RGB以进行一致的最终保存：
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 保存单个帧
        frame_path = os.path.join(out_dir, f"frame_{s:04d}.png")
        # 转换为BGR以用于OpenCV imwrite
        frame_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)

        frame_images.append(img_rgb)

    # 仅在save_grid为True时创建和保存网格图像
    if save_grid:
        # 计算网格尺寸
        num_rows = (S + frames_per_row - 1) // frames_per_row  # 向上取整除法

        # 创建图像网格
        grid_img = None
        for row in range(num_rows):
            start_idx = row * frames_per_row
            end_idx = min(start_idx + frames_per_row, S)

            # 水平连接这一行
            row_img = np.concatenate(frame_images[start_idx:end_idx], axis=1)

            # 如果这一行的图像少于frames_per_row，用黑色填充
            if end_idx - start_idx < frames_per_row:
                padding_width = (frames_per_row - (end_idx - start_idx)) * W
                padding = np.zeros((H, padding_width, 3), dtype=np.uint8)
                row_img = np.concatenate([row_img, padding], axis=1)

            # 将这一行添加到网格中
            if grid_img is None:
                grid_img = row_img
            else:
                grid_img = np.concatenate([grid_img, row_img], axis=0)

        out_path = os.path.join(out_dir, "tracks_grid.png")
        # 转换回BGR以用于OpenCV imwrite
        grid_img_bgr = cv2.cvtColor(grid_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(out_path, grid_img_bgr)
        print(f"[信息] 保存按XY着色的轨迹可视化网格 -> {out_path}")

    print(f"[信息] 保存了{S}个单独的帧到 {out_dir}/frame_*.png")