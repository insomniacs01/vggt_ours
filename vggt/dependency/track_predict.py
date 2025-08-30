# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np
from .vggsfm_utils import *


def predict_tracks(
    images,
    conf=None,
    points_3d=None,
    masks=None,
    max_query_pts=2048,
    query_frame_num=5,
    keypoint_extractor="aliked+sp",
    max_points_num=163840,
    fine_tracking=True,
    complete_non_vis=True,
):
    """
    预测给定图像和掩码的轨迹。

    待办：支持非正方形图像
    待办：支持掩码


    该函数使用指定的查询方法和轨迹预测器预测给定图像和掩码的轨迹。
    它找到查询点，并预测查询帧的轨迹、可见性和分数。

    参数:
        images: 形状为[S, 3, H, W]的输入图像张量。
        conf: 形状为[S, 1, H, W]的置信度分数张量。默认为None。
        points_3d: 包含3D点的张量。默认为None。
        masks: 形状为[S, 1, H, W]的可选掩码张量。默认为None。
        max_query_pts: 最大查询点数。默认为2048。
        query_frame_num: 要使用的查询帧数。默认为5。
        keypoint_extractor: 关键点提取方法。默认为"aliked+sp"。
        max_points_num: 一次处理的最大点数。默认为163840。
        fine_tracking: 是否使用精细跟踪。默认为True。
        complete_non_vis: 是否增强非可见帧。默认为True。

    返回:
        pred_tracks: 包含预测轨迹的Numpy数组。
        pred_vis_scores: 包含轨迹可见性分数的Numpy数组。
        pred_confs: 包含轨迹置信度分数的Numpy数组。
        pred_points_3d: 包含轨迹3D点的Numpy数组。
        pred_colors: 包含轨迹点颜色的Numpy数组。(0, 255)
    """

    device = images.device
    dtype = images.dtype
    tracker = build_vggsfm_tracker().to(device, dtype)

    # 查找查询帧
    query_frame_indexes = generate_rank_by_dino(images, query_frame_num=query_frame_num, device=device)

    # 如果第一张图像不在前面，则将其添加到前面
    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    # 待办：添加处理掩码的功能
    keypoint_extractors = initialize_feature_extractors(
        max_query_pts, extractor_method=keypoint_extractor, device=device
    )

    pred_tracks = []
    pred_vis_scores = []
    pred_confs = []
    pred_points_3d = []
    pred_colors = []

    fmaps_for_tracker = tracker.process_images_to_fmaps(images)

    if fine_tracking:
        print("为了更快的推理，请考虑禁用fine_tracking")

    for query_index in query_frame_indexes:
        print(f"正在预测查询帧 {query_index} 的轨迹")
        pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = _forward_on_query(
            query_index,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            device,
        )

        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        pred_confs.append(pred_conf)
        pred_points_3d.append(pred_point_3d)
        pred_colors.append(pred_color)

    if complete_non_vis:
        pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors = _augment_non_visible_frames(
            pred_tracks,
            pred_vis_scores,
            pred_confs,
            pred_points_3d,
            pred_colors,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            min_vis=500,
            non_vis_thresh=0.1,
            device=device,
        )

    pred_tracks = np.concatenate(pred_tracks, axis=1)
    pred_vis_scores = np.concatenate(pred_vis_scores, axis=1)
    pred_confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
    pred_points_3d = np.concatenate(pred_points_3d, axis=0) if pred_points_3d else None
    pred_colors = np.concatenate(pred_colors, axis=0) if pred_colors else None

    # from vggt.utils.visual_track import visualize_tracks_on_images
    # visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(pred_vis_scores[None])>0.2, out_dir="track_visuals")

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors


def _forward_on_query(
    query_index,
    images,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num,
    fine_tracking,
    device,
):
    """
    处理单个查询帧进行轨迹预测。

    参数:
        query_index: 查询帧的索引
        images: 形状为[S, 3, H, W]的输入图像张量
        conf: 置信度张量
        points_3d: 3D点张量
        fmaps_for_tracker: 跟踪器的特征图
        keypoint_extractors: 初始化的特征提取器
        tracker: VGG-SFM跟踪器
        max_points_num: 一次处理的最大点数
        fine_tracking: 是否使用精细跟踪
        device: 用于计算的设备

    返回:
        pred_track: 预测的轨迹
        pred_vis: 轨迹的可见性分数
        pred_conf: 轨迹的置信度分数
        pred_point_3d: 轨迹的3D点
        pred_color: 轨迹的点颜色 (0, 255)
    """
    frame_num, _, height, width = images.shape

    query_image = images[query_index]
    query_points = extract_keypoints(query_image, keypoint_extractors, round_keypoints=False)
    query_points = query_points[:, torch.randperm(query_points.shape[1], device=device)]

    # 提取关键点位置的颜色
    query_points_long = query_points.squeeze(0).round().long()
    pred_color = images[query_index][:, query_points_long[:, 1], query_points_long[:, 0]]
    pred_color = (pred_color.permute(1, 0).cpu().numpy() * 255).astype(np.uint8)

    # 在关键点位置查询置信度和points_3d
    if (conf is not None) and (points_3d is not None):
        assert height == width
        assert conf.shape[-2] == conf.shape[-1]
        assert conf.shape[:3] == points_3d.shape[:3]
        scale = conf.shape[-1] / width

        query_points_scaled = (query_points.squeeze(0) * scale).round().long()
        query_points_scaled = query_points_scaled.cpu().numpy()

        pred_conf = conf[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]
        pred_point_3d = points_3d[query_index][query_points_scaled[:, 1], query_points_scaled[:, 0]]

        # 启发式地移除低置信度点
        # 我应该将其导出为输入参数吗？
        valid_mask = pred_conf > 1.2
        if valid_mask.sum() > 512:
            query_points = query_points[:, valid_mask]  # 确保形状兼容
            pred_conf = pred_conf[valid_mask]
            pred_point_3d = pred_point_3d[valid_mask]
            pred_color = pred_color[valid_mask]
    else:
        pred_conf = None
        pred_point_3d = None

    reorder_index = calculate_index_mappings(query_index, frame_num, device=device)

    images_feed, fmaps_feed = switch_tensor_order([images, fmaps_for_tracker], reorder_index, dim=0)
    images_feed = images_feed[None]  # 添加批维度
    fmaps_feed = fmaps_feed[None]  # 添加批维度

    all_points_num = images_feed.shape[1] * query_points.shape[1]

    # 不要害怕，这只是分块以让GPU满意
    if all_points_num > max_points_num:
        num_splits = (all_points_num + max_points_num - 1) // max_points_num
        query_points = torch.chunk(query_points, num_splits, dim=1)
    else:
        query_points = [query_points]

    pred_track, pred_vis, _ = predict_tracks_in_chunks(
        tracker, images_feed, query_points, fmaps_feed, fine_tracking=fine_tracking
    )

    pred_track, pred_vis = switch_tensor_order([pred_track, pred_vis], reorder_index, dim=1)

    pred_track = pred_track.squeeze(0).float().cpu().numpy()
    pred_vis = pred_vis.squeeze(0).float().cpu().numpy()

    return pred_track, pred_vis, pred_conf, pred_point_3d, pred_color


def _augment_non_visible_frames(
    pred_tracks: list,  # ← 运行中的np.ndarrays列表
    pred_vis_scores: list,  # ← 运行中的np.ndarrays列表
    pred_confs: list,  # ← 运行中的置信度分数np.ndarrays列表
    pred_points_3d: list,  # ← 运行中的3D点np.ndarrays列表
    pred_colors: list,  # ← 运行中的颜色np.ndarrays列表
    images: torch.Tensor,
    conf,
    points_3d,
    fmaps_for_tracker,
    keypoint_extractors,
    tracker,
    max_points_num: int,
    fine_tracking: bool,
    *,
    min_vis: int = 500,
    non_vis_thresh: float = 0.1,
    device: torch.device = None,
):
    """
    为可见性不足的帧增强跟踪。

    参数:
        pred_tracks: 包含预测轨迹的numpy数组列表。
        pred_vis_scores: 包含可见性分数的numpy数组列表。
        pred_confs: 包含置信度分数的numpy数组列表。
        pred_points_3d: 包含3D点的numpy数组列表。
        pred_colors: 包含点颜色的numpy数组列表。
        images: 形状为[S, 3, H, W]的输入图像张量。
        conf: 形状为[S, 1, H, W]的置信度分数张量
        points_3d: 包含3D点的张量
        fmaps_for_tracker: 跟踪器的特征图
        keypoint_extractors: 初始化的特征提取器
        tracker: VGG-SFM跟踪器
        max_points_num: 一次处理的最大点数
        fine_tracking: 是否使用精细跟踪
        min_vis: 最小可见性阈值
        non_vis_thresh: 非可见性阈值
        device: 用于计算的设备

    返回:
        更新的pred_tracks、pred_vis_scores、pred_confs、pred_points_3d和pred_colors列表。
    """
    last_query = -1
    final_trial = False
    cur_extractors = keypoint_extractors  # 可能在最终尝试时被替换

    while True:
        # 每帧的可见性
        vis_array = np.concatenate(pred_vis_scores, axis=1)

        # 使用numpy计算具有足够可见性的帧
        sufficient_vis_count = (vis_array > non_vis_thresh).sum(axis=-1)
        non_vis_frames = np.where(sufficient_vis_count < min_vis)[0].tolist()

        if len(non_vis_frames) == 0:
            break

        print("正在处理非可见帧：", non_vis_frames)

        # 决定本轮的帧和提取器
        if non_vis_frames[0] == last_query:
            # 同一帧失败两次 - 最终"全力以赴"尝试
            final_trial = True
            cur_extractors = initialize_feature_extractors(2048, extractor_method="sp+sift+aliked", device=device)
            query_frame_list = non_vis_frames  # 一次性处理所有
        else:
            query_frame_list = [non_vis_frames[0]]  # 一次处理一个

        last_query = non_vis_frames[0]

        # 为每个选定的帧运行跟踪器
        for query_index in query_frame_list:
            new_track, new_vis, new_conf, new_point_3d, new_color = _forward_on_query(
                query_index,
                images,
                conf,
                points_3d,
                fmaps_for_tracker,
                cur_extractors,
                tracker,
                max_points_num,
                fine_tracking,
                device,
            )
            pred_tracks.append(new_track)
            pred_vis_scores.append(new_vis)
            pred_confs.append(new_conf)
            pred_points_3d.append(new_point_3d)
            pred_colors.append(new_color)

        if final_trial:
            break  # 最终尝试后停止

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors