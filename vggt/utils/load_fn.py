# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from PIL import Image
from torchvision import transforms as TF
import numpy as np


def load_and_preprocess_images_square(image_path_list, target_size=1024):
    """
    通过中心填充至正方形并调整大小至目标尺寸来加载和预处理图像。
    同时返回原始像素在转换后的位置信息。

    参数:
        image_path_list (list): 图像文件路径列表
        target_size (int, optional): 宽度和高度的目标尺寸。默认为518。

    返回:
        tuple: (
            torch.Tensor: 预处理后的批量图像张量，形状为 (N, 3, target_size, target_size),
            torch.Tensor: 形状为 (N, 5) 的数组，包含每张图像的 [x1, y1, x2, y2, width, height]
        )

    异常:
        ValueError: 如果输入列表为空
    """
    # 检查是否为空列表
    if len(image_path_list) == 0:
        raise ValueError("至少需要1张图像")

    images = []
    original_coords = []  # 重命名为更具描述性的名称
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        # 打开图像
        img = Image.open(image_path)

        # 如果有alpha通道，混合到白色背景上
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)

        # 转换为RGB
        img = img.convert("RGB")

        # 获取原始尺寸
        width, height = img.size

        # 通过填充较短的维度使图像成为正方形
        max_dim = max(width, height)

        # 计算填充
        left = (max_dim - width) // 2
        top = (max_dim - height) // 2

        # 计算调整大小的缩放因子
        scale = target_size / max_dim

        # 计算原始图像在目标空间中的最终坐标
        x1 = left * scale
        y1 = top * scale
        x2 = (left + width) * scale
        y2 = (top + height) * scale

        # 存储原始图像坐标和缩放
        original_coords.append(np.array([x1, y1, x2, y2, width, height]))

        # 创建一个新的黑色正方形图像并粘贴原始图像
        square_img = Image.new("RGB", (max_dim, max_dim), (0, 0, 0))
        square_img.paste(img, (left, top))

        # 调整大小至目标尺寸
        square_img = square_img.resize((target_size, target_size), Image.Resampling.BICUBIC)

        # 转换为张量
        img_tensor = to_tensor(square_img)
        images.append(img_tensor)

    # 堆叠所有图像
    images = torch.stack(images)
    original_coords = torch.from_numpy(np.array(original_coords)).float()

    # 如果是单张图像，添加额外维度以确保正确的形状
    if len(image_path_list) == 1:
        if images.dim() == 3:
            images = images.unsqueeze(0)
            original_coords = original_coords.unsqueeze(0)

    return images, original_coords


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    快速启动函数，用于加载和预处理图像以供模型输入。
    这假设图像应具有相同的形状以便于批处理，但我们的模型也可以很好地处理不同形状的图像。

    参数:
        image_path_list (list): 图像文件路径列表
        mode (str, optional): 预处理模式，"crop" 或 "pad"。
                             - "crop" (默认): 将宽度设置为518px，如果需要则中心裁剪高度。
                             - "pad": 通过使最大维度为518px来保留所有像素，
                               并填充较小的维度以达到正方形形状。

    返回:
        torch.Tensor: 预处理后的批量图像张量，形状为 (N, 3, H, W)

    异常:
        ValueError: 如果输入列表为空或模式无效

    注意:
        - 不同尺寸的图像将用白色填充（值=1.0）
        - 当图像有不同形状时会打印警告
        - 当 mode="crop" 时：函数确保宽度=518px，同时保持纵横比，
          如果高度大于518px，则进行中心裁剪
        - 当 mode="pad" 时：函数确保最大维度为518px，同时保持纵横比，
          较小的维度被填充以达到正方形形状（518x518）
        - 尺寸被调整为可被14整除，以兼容模型要求
    """
    # 检查是否为空列表
    if len(image_path_list) == 0:
        raise ValueError("至少需要1张图像")

    # 验证模式
    if mode not in ["crop", "pad"]:
        raise ValueError("模式必须是 'crop' 或 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # 首先处理所有图像并收集它们的形状
    for image_path in image_path_list:
        # 打开图像
        img = Image.open(image_path)

        # 如果有alpha通道，混合到白色背景上：
        if img.mode == "RGBA":
            # 创建白色背景
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha合成到白色背景上
            img = Image.alpha_composite(background, img)

        # 现在转换为"RGB"（此步骤为透明区域分配白色）
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # 使最大维度为518px，同时保持纵横比
            if width >= height:
                new_width = target_size
                new_height = round(height * (new_width / width) / 14) * 14  # 使其可被14整除
            else:
                new_height = target_size
                new_width = round(width * (new_height / height) / 14) * 14  # 使其可被14整除
        else:  # mode == "crop"
            # 原始行为：将宽度设置为518px
            new_width = target_size
            # 计算保持纵横比的高度，可被14整除
            new_height = round(height * (new_width / width) / 14) * 14

        # 使用新尺寸调整大小 (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # 转换为张量 (0, 1)

        # 如果高度大于518，则中心裁剪高度（仅在裁剪模式下）
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # 对于pad模式，填充以制作target_size x target_size的正方形
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # 用白色填充（值=1.0）
                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # 检查是否有不同的形状
    # 理论上我们的模型也可以很好地处理不同形状的图像
    if len(shapes) > 1:
        print(f"警告：发现具有不同形状的图像：{shapes}")
        # 找出最大尺寸
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # 如有必要，填充图像
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img, (pad_left, pad_right, pad_top, pad_bottom), mode="constant", value=1.0
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # 连接图像

    # 确保单张图像时的正确形状
    if len(image_path_list) == 1:
        # 验证形状是 (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images