# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Callable, Optional

from hydra.utils import instantiate
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, Sampler
from abc import ABC, abstractmethod
from training.data.composed_dataset import ComposedDataset
from .worker_fn import get_worker_init_fn

def custom_detection_collate(batch_list):
    if not batch_list:
        return {}

    result = {}
    batch_size = len(batch_list)

    for key in batch_list[0].keys():
        if key in ['gt_boxes', 'gt_classes']:
            # GT数据保持为列表格式
            gt_list = []
            for item in batch_list:
                data = item[key]
                if not torch.is_tensor(data):
                    data = torch.tensor(data,
                                        dtype=torch.float32 if key == 'gt_boxes' else torch.long)
                gt_list.append(data)
            result[key] = gt_list

        elif key == 'seq_name':
            # 字符串列表
            result[key] = [item[key] for item in batch_list]

        elif key in ['ids', 'original_sizes']:
            # 这些键可能有不同的形状，需要特殊处理
            all_items = []
            for item in batch_list:
                data = item[key]
                if not torch.is_tensor(data):
                    data = torch.tensor(data)
                all_items.append(data)

            # 检查是否所有张量形状相同
            shapes = [t.shape for t in all_items]
            if len(set(shapes)) == 1:
                # 形状相同，可以stack
                result[key] = torch.stack(all_items)
            else:
                # 形状不同，保持为列表
                result[key] = all_items

        elif key in ['images', 'depths', 'extrinsics', 'intrinsics',
                     'cam_points', 'world_points', 'point_masks']:
            # 这些是per-frame的数据，需要特殊处理
            all_items = []
            for item in batch_list:
                data = item[key]
                if not torch.is_tensor(data):
                    data = torch.tensor(data)
                all_items.append(data)

            # 检查序列长度
            seq_lengths = [t.shape[0] for t in all_items]

            if len(set(seq_lengths)) == 1:
                # 所有序列长度相同，可以stack
                result[key] = torch.stack(all_items)
            else:
                # 序列长度不同，需要padding或保持为列表
                max_seq_len = max(seq_lengths)

                # 对于图像相关数据，进行padding
                if key in ['images', 'depths', 'cam_points', 'world_points', 'point_masks']:
                    padded_items = []
                    for i, item in enumerate(all_items):
                        seq_len = item.shape[0]
                        if seq_len < max_seq_len:
                            # 计算padding
                            pad_size = max_seq_len - seq_len
                            pad_shape = [pad_size] + list(item.shape[1:])

                            # 创建padding（使用zeros）
                            if key == 'images':
                                padding = torch.zeros(pad_shape, dtype=item.dtype)
                            elif key == 'depths':
                                padding = torch.zeros(pad_shape, dtype=item.dtype)
                            elif key == 'point_masks':
                                padding = torch.zeros(pad_shape, dtype=torch.bool)
                            else:
                                padding = torch.zeros(pad_shape, dtype=item.dtype)

                            # 连接
                            item = torch.cat([item, padding], dim=0)
                        padded_items.append(item)
                    result[key] = torch.stack(padded_items)

                    # 添加mask来标记有效帧
                    if 'valid_frames_mask' not in result:
                        valid_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
                        for i, seq_len in enumerate(seq_lengths):
                            valid_mask[i, :seq_len] = True
                        result['valid_frames_mask'] = valid_mask
                else:
                    # 对于外参和内参，保持为列表
                    result[key] = all_items
        else:
            # 其他键的默认处理
            if torch.is_tensor(batch_list[0][key]):
                try:
                    result[key] = torch.stack([item[key] for item in batch_list])
                except RuntimeError:
                    # 如果stack失败，保持为列表
                    result[key] = [item[key] for item in batch_list]
            else:
                result[key] = [item[key] for item in batch_list]

    return result

class DynamicTorchDataset(ABC):
    def __init__(
            self,
            dataset: dict,
            common_config: dict,
            num_workers: int,
            shuffle: bool,
            pin_memory: bool,
            drop_last: bool = True,
            collate_fn: Optional[Callable] = None,
            worker_init_fn: Optional[Callable] = None,
            persistent_workers: bool = False,
            seed: int = 42,
            max_img_per_gpu: int = 48,
    ) -> None:
        self.dataset_config = dataset
        self.common_config = common_config
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.max_img_per_gpu = max_img_per_gpu

        # Instantiate the dataset
        self.dataset = ComposedDataset(
            dataset_configs=dataset['dataset_configs'],
            common_config=common_config
        )

        # Extract aspect ratio and image number ranges from the configuration
        self.aspect_ratio_range = common_config.augs.aspects  # e.g., [0.5, 1.0]
        self.image_num_range = common_config.img_nums  # e.g., [2, 24]

        # Validate the aspect ratio and image number ranges
        if len(self.aspect_ratio_range) != 2 or self.aspect_ratio_range[0] > self.aspect_ratio_range[1]:
            raise ValueError(f"aspect_ratio_range must be [min, max] with min <= max, got {self.aspect_ratio_range}")
        if len(self.image_num_range) != 2 or self.image_num_range[0] < 1 or self.image_num_range[0] > \
                self.image_num_range[1]:
            raise ValueError(f"image_num_range must be [min, max] with 1 <= min <= max, got {self.image_num_range}")

        # Create samplers
        self.sampler = DynamicDistributedSampler(self.dataset, seed=seed, shuffle=shuffle)
        self.batch_sampler = DynamicBatchSampler(
            self.sampler,
            self.aspect_ratio_range,
            self.image_num_range,
            seed=seed,
            max_img_per_gpu=max_img_per_gpu
        )
        if collate_fn is None:
            collate_fn = custom_detection_collate
        self.collate_fn = collate_fn

    def get_loader(self, epoch):
        print("Building dynamic dataloader with epoch:", epoch)

        # Set the epoch for the sampler
        self.sampler.set_epoch(epoch)
        if hasattr(self.dataset, "epoch"):
            self.dataset.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

        # Create and return the dataloader
        return DataLoader(
            self.dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_sampler=self.batch_sampler,
            collate_fn=self.collate_fn,
            persistent_workers=self.persistent_workers,
            worker_init_fn=get_worker_init_fn(
                seed=self.seed,
                num_workers=self.num_workers,
                epoch=epoch,
                worker_init_fn=self.worker_init_fn,
            ),
        )

class DynamicBatchSampler(Sampler):
    """
    A custom batch sampler that dynamically adjusts batch size, aspect ratio, and image number
    for each sample. Batches within a sample share the same aspect ratio and image number.
    """

    def __init__(self,
                 sampler,
                 aspect_ratio_range,
                 image_num_range,
                 epoch=0,
                 seed=42,
                 max_img_per_gpu=48):
        """
        Initializes the dynamic batch sampler.

        Args:
            sampler: Instance of DynamicDistributedSampler.
            aspect_ratio_range: List containing [min_aspect_ratio, max_aspect_ratio].
            image_num_range: List containing [min_images, max_images] per sample.
            epoch: Current epoch number.
            seed: Random seed for reproducibility.
            max_img_per_gpu: Maximum number of images to fit in GPU memory.
        """
        self.sampler = sampler
        self.aspect_ratio_range = aspect_ratio_range
        self.image_num_range = image_num_range
        self.rng = random.Random()

        # Uniformly sample from the range of possible image numbers
        # For any image number, the weight is 1.0 (uniform sampling). You can set any different weights here.
        self.image_num_weights = {num_images: 1.0 for num_images in range(image_num_range[0], image_num_range[1] + 1)}

        # Possible image numbers, e.g., [2, 3, 4, ..., 24]
        self.possible_nums = np.array([n for n in self.image_num_weights.keys()
                                       if self.image_num_range[0] <= n <= self.image_num_range[1]])

        # Normalize weights for sampling
        weights = [self.image_num_weights[n] for n in self.possible_nums]
        self.normalized_weights = np.array(weights) / sum(weights)

        # Maximum image number per GPU
        self.max_img_per_gpu = max_img_per_gpu

        # Set the epoch for the sampler
        self.set_epoch(epoch + seed)

    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler, affecting the random sequence.

        Args:
            epoch: The epoch number.
        """
        self.sampler.set_epoch(epoch)
        self.epoch = epoch
        self.rng.seed(epoch * 100)

    def __iter__(self):
        """
        Yields batches of samples with synchronized dynamic parameters.

        Returns:
            Iterator yielding batches of indices with associated parameters.
        """
        sampler_iterator = iter(self.sampler)

        while True:
            try:
                # Sample random image number and aspect ratio
                random_image_num = int(np.random.choice(self.possible_nums, p=self.normalized_weights))
                random_aspect_ratio = round(self.rng.uniform(self.aspect_ratio_range[0], self.aspect_ratio_range[1]), 2)

                # Update sampler parameters
                self.sampler.update_parameters(
                    aspect_ratio=random_aspect_ratio,
                    image_num=random_image_num
                )

                # Calculate batch size based on max images per GPU and current image number
                batch_size = self.max_img_per_gpu / random_image_num
                batch_size = np.floor(batch_size).astype(int)
                batch_size = max(1, batch_size)  # Ensure batch size is at least 1

                # Collect samples for the current batch
                current_batch = []
                for _ in range(batch_size):
                    try:
                        item = next(sampler_iterator)  # item is (idx, aspect_ratio, image_num)
                        current_batch.append(item)
                    except StopIteration:
                        break  # No more samples

                if not current_batch:
                    break  # No more data to yield

                yield current_batch

            except StopIteration:
                break  # End of sampler's iterator

    def __len__(self):
        # Return a large dummy length
        return 1000000


class DynamicDistributedSampler(DistributedSampler):
    """
    Extends PyTorch's DistributedSampler to include dynamic aspect_ratio and image_num
    parameters, which can be passed into the dataset's __getitem__ method.
    """

    def __init__(
            self,
            dataset,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = False,
            seed: int = 0,
            drop_last: bool = False,
    ):
        import torch.distributed as dist

        if num_replicas is None:
            if dist.is_available() and dist.is_initialized():
                num_replicas = dist.get_world_size()
            else:
                num_replicas = 1

        if rank is None:
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0

        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.aspect_ratio = None
        self.image_num = None

    def __iter__(self):
        """
        Yields a sequence of (index, image_num, aspect_ratio).
        Relies on the parent class's logic for shuffling/distributing
        the indices across replicas, then attaches extra parameters.
        """
        indices_iter = super().__iter__()

        for idx in indices_iter:
            yield (idx, self.image_num, self.aspect_ratio,)

    def update_parameters(self, aspect_ratio, image_num):
        """
        Updates dynamic parameters for each new epoch or iteration.

        Args:
            aspect_ratio: The aspect ratio to set.
            image_num: The number of images to set.
        """
        self.aspect_ratio = aspect_ratio
        self.image_num = image_num