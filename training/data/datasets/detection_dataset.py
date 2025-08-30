import torch
import numpy as np
from pathlib import Path
import yaml
from PIL import Image
import logging
import random

from ..base_dataset import BaseDataset
from ..dataset_util import *


class Detection3DDataset(BaseDataset):
    def __init__(
            self,
            common_conf,
            split: str = "train",
            data_dirs=None,
            max_objects: int = 100,
            len_train: int = 160,
            len_test: int = 40,
            train_ratio: float = 0.8,
    ):
        super().__init__(common_conf=common_conf)

        # 从common_conf获取通用设置
        self.training = common_conf.training
        self.load_depth = getattr(common_conf, 'load_depth', False)
        self.inside_random = getattr(common_conf, 'inside_random', False)
        self.allow_duplicate_img = getattr(common_conf, 'allow_duplicate_img', True)

        if data_dirs is None:
            data_dirs = ['../datasets/detection_3d/train', '../3242']
        elif isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        self.split = split
        self.max_objects = max_objects
        self.samples = []

        # 加载数据样本
        for data_dir_str in data_dirs:
            data_dir = Path(data_dir_str).expanduser()
            if not data_dir.exists():
                logging.warning(f"Data directory does not exist: {data_dir}")
                continue

            if self._is_format1(data_dir):
                self._load_format1(data_dir, split, len_train, len_test)
            else:
                self._load_format2(data_dir, split, train_ratio)

        # 设置数据集长度
        if split == "train":
            self.len_train = len(self.samples)
        else:
            self.len_train = len(self.samples)

        # 类别映射
        self.class_map = {
            'Car': 0, 'Truck': 1, 'Bus': 2, 'Pedestrian': 3,
            'TrashCan': 4, 'Cyclist': 5, 'Motorcycle': 6,
            'BicycleRider': 7, 'MotorcyleRider': 8, 'ConcreteTruck': 9,
            'Vehicle': 0,
        }

        logging.info(f"Detection3DDataset initialized: {len(self.samples)} {split} samples")
        logging.info(f"Dataset length set to: {self.len_train}")

    def _is_format1(self, data_dir):
        """检测是否是格式1（cam1.jpeg, cam2.jpeg）"""
        test_files = list(data_dir.glob("*_cam1.jpeg"))
        return len(test_files) > 0

    def _load_format1(self, data_dir, split, len_train, len_test):
        """加载格式1的数据"""
        yaml_files = sorted(data_dir.glob("*.yaml"))

        if split == 'train':
            yaml_files = yaml_files[:len_train]
        else:
            yaml_files = yaml_files[len_train:len_train + len_test]

        for yaml_file in yaml_files:
            base_name = yaml_file.stem
            camera_paths = []
            for i in range(1, 10):
                cam_path = data_dir / f"{base_name}_cam{i}.jpeg"
                if cam_path.exists():
                    camera_paths.append(cam_path)
                else:
                    break

            if camera_paths:
                self.samples.append({
                    'id': base_name,
                    'yaml': yaml_file,
                    'cameras': camera_paths,
                    'format': 1
                })

    def _load_format2(self, data_dir, split, train_ratio):
        """加载格式2的数据"""
        yaml_files = sorted(data_dir.glob("*.yaml"))
        split_idx = int(len(yaml_files) * train_ratio)

        if split == 'train':
            yaml_files = yaml_files[:split_idx]
        else:
            yaml_files = yaml_files[split_idx:]

        for yaml_file in yaml_files:
            base_name = yaml_file.stem
            camera_paths = []
            for i in range(10):
                for ext in ['.png', '.jpg', '.jpeg']:
                    cam_path = data_dir / f"{base_name}_camera{i}{ext}"
                    if cam_path.exists():
                        camera_paths.append(cam_path)
                        break

            if camera_paths:
                self.samples.append({
                    'id': base_name,
                    'yaml': yaml_file,
                    'cameras': camera_paths,
                    'format': 2
                })

    def get_data(
            self,
            seq_index: int = None,
            img_per_seq: int = None,
            seq_name: str = None,
            ids: list = None,
            aspect_ratio: float = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.
        """
        if self.inside_random:
            seq_index = random.randint(0, len(self.samples) - 1)

        if seq_name is None:
            sample = self.samples[seq_index]
        else:
            sample = None
            for s in self.samples:
                if s['id'] == seq_name:
                    sample = s
                    break
            if sample is None:
                raise ValueError(f"Sequence {seq_name} not found")

        # 加载YAML标注
        with open(sample['yaml'], 'r') as f:
            anno = yaml.safe_load(f)

        # 获取目标图像形状
        target_image_shape = self.get_target_shape(aspect_ratio)

        # 解析车辆标注和相机参数
        if sample['format'] == 1:
            gt_boxes, gt_classes = self._parse_vehicles_format1(anno)
            all_extrinsics, all_intrinsics = self._parse_cameras_format1(anno, len(sample['cameras']))
        else:
            gt_boxes, gt_classes = self._parse_vehicles_format2(anno)
            all_extrinsics, all_intrinsics = self._parse_cameras_format2(anno, len(sample['cameras']))

        # 确定要加载的图像ID
        num_cameras = len(sample['cameras'])

        if ids is None:
            if img_per_seq is None:
                img_per_seq = num_cameras

            if img_per_seq > num_cameras:
                # 如果请求的图像多于可用相机，使用重复采样
                ids = np.random.choice(num_cameras, img_per_seq, replace=True)
            else:
                # 否则，不重复采样
                ids = np.random.choice(num_cameras, img_per_seq, replace=False)

        # 处理每个相机图像
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        image_paths = []
        original_sizes = []

        for cam_id in ids:
            cam_path = sample['cameras'][cam_id]
            image = read_image_cv2(str(cam_path))

            if image is None:
                image = np.zeros((target_image_shape[0], target_image_shape[1], 3), dtype=np.uint8)

            # 创建深度占位符
            original_size = np.array(image.shape[:2])
            depth_map = np.zeros(image.shape[:2], dtype=np.float32)

            # 获取对应的相机参数
            extri_opencv = all_extrinsics[cam_id] if cam_id < len(all_extrinsics) else np.eye(3, 4, dtype=np.float32)
            intri_opencv = all_intrinsics[cam_id] if cam_id < len(all_intrinsics) else np.array(
                [[500, 0, 259], [0, 500, 259], [0, 0, 1]], dtype=np.float32)

            # 使用BaseDataset的图像处理方法
            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=str(cam_path),
            )

            images.append(image)
            if depth_map is not None and len(depth_map.shape) == 2:
                depths.append(depth_map)
            else:
                depths.append(np.zeros(image.shape[:2], dtype=np.float32))
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points if cam_coords_points is not None else np.zeros(image.shape[:2] + (3,)))
            world_points.append(
                world_coords_points if world_coords_points is not None else np.zeros(image.shape[:2] + (3,)))
            point_masks.append(point_mask if point_mask is not None else np.zeros(image.shape[:2], dtype=bool))
            image_paths.append(str(cam_path))
            original_sizes.append(original_size)

        # GT数据应该是numpy数组
        if len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_classes = np.zeros((0,), dtype=np.int64)
        else:
            gt_boxes = np.array(gt_boxes, dtype=np.float32)
            gt_classes = np.array(gt_classes, dtype=np.int64)

        # 构建batch
        batch = {
            "seq_name": "detection3d_" + sample['id'],
            "ids": np.array(ids),  # 确保是numpy数组
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
            "tracks": None,
            "track_masks": None,
            "gt_boxes": gt_boxes,  # numpy array [N, 7]
            "gt_classes": gt_classes,  # numpy array [N]
        }

        return batch

    def _parse_vehicles_format1(self, anno):
        """解析格式1的3D框标注"""
        boxes = []
        classes = []
        ego_pose = anno.get('true_ego_pose', [0, 0, 0, 0, 0, 0])
        ego_position = ego_pose[:3]
        ego_yaw = ego_pose[5]
        vehicles = anno.get('vehicles', {})

        for vid, vdata in list(vehicles.items())[:self.max_objects]:
            world_loc = vdata.get('location', [0, 0, 0])
            extent = vdata.get('extent', [2.0, 1.0, 1.5])
            angle = vdata.get('angle', [0, 0, 0])
            obj_yaw = angle[-1] if isinstance(angle, list) else angle

            relative_loc = self._world_to_ego(world_loc, ego_position, ego_yaw)
            relative_yaw = obj_yaw - ego_yaw
            box = relative_loc + extent + [relative_yaw]
            boxes.append(box)

            obj_type = vdata.get('obj_type', 'Car')
            cls = self.class_map.get(obj_type, 0)
            classes.append(cls)

        return boxes, classes  # 返回列表，会在get_data中转换为numpy数组

    def _parse_vehicles_format2(self, anno):
        """解析格式2的3D框标注"""
        boxes = []
        classes = []
        ego_pose = anno.get('true_ego_pos', [0, 0, 0, 0, 0, 0])
        if isinstance(ego_pose, list):
            ego_position = ego_pose[:3]
            ego_yaw = np.deg2rad(ego_pose[5]) if len(ego_pose) > 5 else 0
        else:
            ego_position = [0, 0, 0]
            ego_yaw = 0

        vehicles = anno.get('vehicles', {})

        for vid, vdata in list(vehicles.items())[:self.max_objects]:
            location = vdata.get('location', [0, 0, 0])
            if isinstance(location, dict):
                world_loc = [location.get('x', 0), location.get('y', 0), location.get('z', 0)]
            else:
                world_loc = location[:3] if isinstance(location, list) else [0, 0, 0]

            extent = vdata.get('extent', [2.0, 1.0, 1.5])
            if isinstance(extent, dict):
                extent = [extent.get('x', 2.0), extent.get('y', 1.0), extent.get('z', 1.5)]

            angle = vdata.get('angle', [0, 0, 0])
            if isinstance(angle, dict):
                obj_yaw = np.deg2rad(angle.get('yaw', 0))
            elif isinstance(angle, list):
                obj_yaw = np.deg2rad(angle[-1]) if len(angle) > 0 else 0
            else:
                obj_yaw = np.deg2rad(float(angle))

            relative_loc = self._world_to_ego(world_loc, ego_position, ego_yaw)
            relative_yaw = obj_yaw - ego_yaw
            box = relative_loc + extent + [relative_yaw]
            boxes.append(box)

            obj_type = vdata.get('obj_type', vdata.get('type', 'Car'))
            cls = self.class_map.get(obj_type, 0)
            classes.append(cls)

        return boxes, classes  # 返回列表，会在get_data中转换为numpy数组

    def _world_to_ego(self, world_pos, ego_pos, ego_yaw):
        """世界坐标转自车坐标"""
        relative = [world_pos[i] - ego_pos[i] for i in range(3)]
        cos_yaw = np.cos(-ego_yaw)
        sin_yaw = np.sin(-ego_yaw)
        x_ego = relative[0] * cos_yaw - relative[1] * sin_yaw
        y_ego = relative[0] * sin_yaw + relative[1] * cos_yaw
        z_ego = relative[2]
        return [x_ego, y_ego, z_ego]

    def _parse_cameras_format1(self, anno, num_cameras):
        """解析格式1的相机参数"""
        extrinsics = []
        intrinsics = []

        for i in range(num_cameras):
            cam_key = f'cam{i + 1}'
            cam_data = anno.get(cam_key, {})
            ext = cam_data.get('extrinsic', [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            extrinsics.append(np.array(ext[:3], dtype=np.float32))
            intr = cam_data.get('intrinsic', [[500, 0, 259], [0, 500, 259], [0, 0, 1]])
            intrinsics.append(np.array(intr, dtype=np.float32))

        return extrinsics, intrinsics

    def _parse_cameras_format2(self, anno, num_cameras):
        """解析格式2的相机参数"""
        extrinsics = []
        intrinsics = []

        for i in range(num_cameras):
            cam_key = f'camera{i}'
            cam_data = anno.get(cam_key, {})
            ext = cam_data.get('extrinsic', [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
            if isinstance(ext, list) and len(ext) >= 3:
                extrinsics.append(np.array(ext[:3], dtype=np.float32))
            else:
                extrinsics.append(np.eye(3, 4, dtype=np.float32))

            intr = cam_data.get('intrinsic', [[500, 0, 259], [0, 500, 259], [0, 0, 1]])
            if isinstance(intr, list) and len(intr) >= 3:
                intrinsics.append(np.array(intr[:3], dtype=np.float32))
            else:
                intrinsics.append(np.array([[500, 0, 259], [0, 500, 259], [0, 0, 1]], dtype=np.float32))

        return extrinsics, intrinsics