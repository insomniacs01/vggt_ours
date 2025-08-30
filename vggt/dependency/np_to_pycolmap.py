# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pycolmap
from .projection import project_3D_points_np


def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
):
    """
    将批量NumPy数组转换为PyCOLMAP格式

    查看 https://github.com/colmap/pycolmap 了解更多关于其格式的详细信息

    注意colmap期望images/cameras/points3D使用1索引
    所以colmap索引和批索引之间有+1的偏移


    注意：与VGGSfM不同，这个函数：
    1. 使用np而不是torch
    2. 帧索引和相机ID从1开始而不是0（以适应PyCOLMAP的格式）
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, 假设所有帧都已填充到相同大小
    # 其中N是帧数，P是轨迹数

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    reproj_mask = None

    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    if masks.sum(1).min() < min_inlier_per_frame:
        print(f"每帧内点不足，跳过BA。")
        return None, None

    # 重建对象，遵循PyCOLMAP/COLMAP的格式
    reconstruction = pycolmap.Reconstruction()

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # 如果没有两个内点，则轨迹无效
    valid_idx = np.nonzero(valid_mask)[0]

    # 只添加具有足够2D点的3D点
    for vidx in valid_idx:
        # 如果提供了RGB颜色则使用，否则使用零
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3)
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), rgb)

    num_points3D = len(valid_idx)
    camera = None
    # 帧索引
    for fidx in range(N):
        # 设置相机
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # 添加相机
            reconstruction.add_camera(camera)

        # 设置图像
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # 旋转和平移

        image = pycolmap.Image(
            id=fidx + 1, name=f"image_{fidx + 1}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        # 注意point3D_id从1开始
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]

            if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all():
                if masks[fidx][original_track_idx]:
                    # 看起来我们不需要为BA加+0.5
                    point2D_xy = tracks[fidx][original_track_idx]
                    # 请注意在添加Point2D对象时
                    # 它不仅需要2D xy位置，还需要到3D点的id
                    points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

                    # 添加元素
                    track = reconstruction.points3D[point3D_id].track
                    track.add_element(fidx + 1, point2D_idx)
                    point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"帧 {fidx + 1} 超出BA范围")
            image.registered = False

        # 添加图像
        reconstruction.add_image(image)

    return reconstruction, valid_mask


def pycolmap_to_batch_np_matrix(reconstruction, device="cpu", camera_type="SIMPLE_PINHOLE"):
    """
    将PyCOLMAP重建对象转换为批量NumPy数组。

    参数:
        reconstruction (pycolmap.Reconstruction): 来自PyCOLMAP的重建对象。
        device (str): 在NumPy版本中被忽略（保留以兼容API）。
        camera_type (str): 使用的相机模型类型（默认："SIMPLE_PINHOLE"）。

    返回:
        tuple: 包含points3D、extrinsics、intrinsics和可选的extra_params的元组。
    """

    num_images = len(reconstruction.images)
    max_points3D_id = max(reconstruction.point3D_ids())
    points3D = np.zeros((max_points3D_id, 3))

    for point3D_id in reconstruction.points3D:
        points3D[point3D_id - 1] = reconstruction.points3D[point3D_id].xyz

    extrinsics = []
    intrinsics = []

    extra_params = [] if camera_type == "SIMPLE_RADIAL" else None

    for i in range(num_images):
        # 提取并添加外参
        pyimg = reconstruction.images[i + 1]
        pycam = reconstruction.cameras[pyimg.camera_id]
        matrix = pyimg.cam_from_world.matrix()
        extrinsics.append(matrix)

        # 提取并添加内参
        calibration_matrix = pycam.calibration_matrix()
        intrinsics.append(calibration_matrix)

        if camera_type == "SIMPLE_RADIAL":
            extra_params.append(pycam.params[-1])

    # 将列表转换为NumPy数组而不是torch张量
    extrinsics = np.stack(extrinsics)
    intrinsics = np.stack(intrinsics)

    if camera_type == "SIMPLE_RADIAL":
        extra_params = np.stack(extra_params)
        extra_params = extra_params[:, None]

    return points3D, extrinsics, intrinsics, extra_params


########################################################


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
):
    """
    将批量NumPy数组转换为PyCOLMAP

    与batch_np_matrix_to_pycolmap不同，这个函数不使用轨迹。

    它只将points3d保存为colmap重建格式，以作为高斯或其他新视图合成方法的初始化。

    不要用于BA。
    """
    # points3d: Px3
    # points_xyf: Px3，包含x、y坐标和帧索引
    # points_rgb: Px3，rgb颜色
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2，假设所有帧都已填充到相同大小
    # 其中N是帧数，P是轨迹数

    N = len(extrinsics)
    P = len(points3d)

    # 重建对象，遵循PyCOLMAP/COLMAP的格式
    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # 帧索引
    for fidx in range(N):
        # 设置相机
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # 添加相机
            reconstruction.add_camera(camera)

        # 设置图像
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # 旋转和平移

        image = pycolmap.Image(
            id=fidx + 1, name=f"image_{fidx + 1}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            # 添加元素
            track = reconstruction.points3D[point3D_id].track
            track.add_element(fidx + 1, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"帧 {fidx + 1} 没有任何点")
            image.registered = False

        # 添加图像
        reconstruction.add_image(image)

    return reconstruction


def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    根据相机类型获取相机参数的辅助函数。

    参数:
        fidx: 帧索引
        intrinsics: 相机内参
        camera_type: 相机模型类型
        extra_params: 某些相机类型的额外参数

    返回:
        pycolmap_intri: 相机参数的NumPy数组
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL尚不支持")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"相机类型 {camera_type} 尚不支持")

    return pycolmap_intri