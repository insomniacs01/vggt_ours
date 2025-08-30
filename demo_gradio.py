# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

print("正在初始化并加载VGGT模型...")
# model = VGGT.from_pretrained("facebook/VGGT-1B")  # 另一种加载模型的方式

model = VGGT()
# _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
# model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model_path = "model.pt"
model.load_state_dict(torch.load(model_path))

model.eval()
model = model.to(device)


# -------------------------------------------------------------------------
# 1) 核心模型推理
# -------------------------------------------------------------------------
def run_model(target_dir, model) -> dict:
    """
    在 'target_dir/images' 文件夹中的图像上运行VGGT模型并返回预测结果。
    """
    print(f"正在处理来自 {target_dir} 的图像")

    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA不可用。请检查你的环境配置。")

    # 将模型移到设备上
    model = model.to(device)
    model.eval()

    # 加载和预处理图像
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"找到 {len(image_names)} 张图像")
    if len(image_names) == 0:
        raise ValueError("未找到图像。请检查你的上传。")

    images = load_and_preprocess_images(image_names).to(device)
    print(f"预处理后的图像形状: {images.shape}")

    # 运行推理
    print("正在运行推理...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    # 将位姿编码转换为外参和内参矩阵
    print("正在将位姿编码转换为外参和内参矩阵...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # 将张量转换为numpy数组
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)  # 移除批次维度
    predictions['pose_enc_list'] = None  # 移除pose_enc_list

    # 从深度图生成世界点
    print("正在从深度图计算世界点...")
    depth_map = predictions["depth"]  # (S, H, W, 1)
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    # 清理内存
    torch.cuda.empty_cache()
    return predictions


# -------------------------------------------------------------------------
# 2) 处理上传的视频/图像 --> 生成target_dir + images
# -------------------------------------------------------------------------
def handle_uploads(input_video, input_images):
    """
    创建一个新的 'target_dir' + 'images' 子文件夹，并将用户上传的
    图像或从视频中提取的帧放入其中。返回 (target_dir, image_paths)。
    """
    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # 创建唯一的文件夹名称
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"input_images_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")

    # 如果文件夹已存在则清理
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir)
    os.makedirs(target_dir_images)

    image_paths = []

    # --- 处理图像 ---
    if input_images is not None:
        for file_data in input_images:
            if isinstance(file_data, dict) and "name" in file_data:
                file_path = file_data["name"]
            else:
                file_path = file_data
            dst_path = os.path.join(target_dir_images, os.path.basename(file_path))
            shutil.copy(file_path, dst_path)
            image_paths.append(dst_path)

    # --- 处理视频 ---
    if input_video is not None:
        if isinstance(input_video, dict) and "name" in input_video:
            video_path = input_video["name"]
        else:
            video_path = input_video

        vs = cv2.VideoCapture(video_path)
        fps = vs.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 1)  # 每秒1帧

        count = 0
        video_frame_num = 0
        while True:
            gotit, frame = vs.read()
            if not gotit:
                break
            count += 1
            if count % frame_interval == 0:
                image_path = os.path.join(target_dir_images, f"{video_frame_num:06}.png")
                cv2.imwrite(image_path, frame)
                image_paths.append(image_path)
                video_frame_num += 1

    # 为画廊排序最终图像
    image_paths = sorted(image_paths)

    end_time = time.time()
    print(f"文件已复制到 {target_dir_images}；耗时 {end_time - start_time:.3f} 秒")
    return target_dir, image_paths


# -------------------------------------------------------------------------
# 3) 上传时更新画廊
# -------------------------------------------------------------------------
def update_gallery_on_upload(input_video, input_images):
    """
    每当用户上传或更改文件时，立即处理它们
    并在画廊中显示。返回 (target_dir, image_paths)。
    如果没有上传任何内容，返回 "None" 和空列表。
    """
    if not input_video and not input_images:
        return None, None, None, None
    target_dir, image_paths = handle_uploads(input_video, input_images)
    return None, target_dir, image_paths, "上传完成。点击'重建'开始3D处理。"


# -------------------------------------------------------------------------
# 4) 重建：使用target_dir加上任何可视化参数
# -------------------------------------------------------------------------
def gradio_demo(
        target_dir,
        conf_thres=3.0,
        frame_filter="All",
        mask_black_bg=False,
        mask_white_bg=False,
        show_cam=True,
        mask_sky=False,
        prediction_mode="点图回归",
):
    """
    使用已创建的target_dir/images执行重建。
    """
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "未找到有效的目标目录。请先上传文件。", None, None

    start_time = time.time()
    gc.collect()
    torch.cuda.empty_cache()

    # 准备帧过滤下拉菜单
    target_dir_images = os.path.join(target_dir, "images")
    all_files = sorted(os.listdir(target_dir_images)) if os.path.isdir(target_dir_images) else []
    all_files = [f"{i}: {filename}" for i, filename in enumerate(all_files)]
    frame_filter_choices = ["全部"] + all_files

    print("正在运行模型...")
    with torch.no_grad():
        predictions = run_model(target_dir, model)

    # 保存预测结果
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)

    # 处理None的frame_filter
    if frame_filter is None:
        frame_filter = "全部"

    # 构建GLB文件名
    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    # 将预测转换为GLB
    glbscene = predictions_to_glb(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        mask_black_bg=mask_black_bg,
        mask_white_bg=mask_white_bg,
        show_cam=show_cam,
        mask_sky=mask_sky,
        target_dir=target_dir,
        prediction_mode=prediction_mode,
    )
    glbscene.export(file_obj=glbfile)

    # 清理内存
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒 (包括IO)")
    log_msg = f"重建成功 ({len(all_files)} 帧)。等待可视化。"

    return glbfile, log_msg, gr.Dropdown(choices=frame_filter_choices, value=frame_filter, interactive=True)


# -------------------------------------------------------------------------
# 5) UI重置和重新可视化的辅助函数
# -------------------------------------------------------------------------
def clear_fields():
    """
    清除3D查看器、存储的target_dir并清空画廊。
    """
    return None


def update_log():
    """
    在等待时显示快速日志消息。
    """
    return "正在加载和重建..."


def update_visualization(
        target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode,
        is_example
):
    """
    从npz重新加载保存的预测，为新参数创建（或重用）GLB，
    并将其返回给3D查看器。如果is_example == "True"，则跳过。
    """

    # 如果是示例点击，按要求跳过
    if is_example == "True":
        return None, "没有可用的重建。请先点击重建按钮。"

    if not target_dir or target_dir == "None" or not os.path.isdir(target_dir):
        return None, "没有可用的重建。请先点击重建按钮。"

    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None, f"在 {predictions_path} 没有可用的重建。请先运行'重建'。"

    key_list = [
        "pose_enc",
        "depth",
        "depth_conf",
        "world_points",
        "world_points_conf",
        "images",
        "extrinsic",
        "intrinsic",
        "world_points_from_depth",
    ]

    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in key_list}

    glbfile = os.path.join(
        target_dir,
        f"glbscene_{conf_thres}_{frame_filter.replace('.', '_').replace(':', '').replace(' ', '_')}_maskb{mask_black_bg}_maskw{mask_white_bg}_cam{show_cam}_sky{mask_sky}_pred{prediction_mode.replace(' ', '_')}.glb",
    )

    if not os.path.exists(glbfile):
        glbscene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames=frame_filter,
            mask_black_bg=mask_black_bg,
            mask_white_bg=mask_white_bg,
            show_cam=show_cam,
            mask_sky=mask_sky,
            target_dir=target_dir,
            prediction_mode=prediction_mode,
        )
        glbscene.export(file_obj=glbfile)

    return glbfile, "正在更新可视化"


# -------------------------------------------------------------------------
# 示例图像
# -------------------------------------------------------------------------

great_wall_video = "examples/videos/great_wall.mp4"
colosseum_video = "examples/videos/Colosseum.mp4"
room_video = "examples/videos/room.mp4"
kitchen_video = "examples/videos/kitchen.mp4"
fern_video = "examples/videos/fern.mp4"
single_cartoon_video = "examples/videos/single_cartoon.mp4"
single_oil_painting_video = "examples/videos/single_oil_painting.mp4"
pyramid_video = "examples/videos/pyramid.mp4"

# -------------------------------------------------------------------------
# 6) 构建Gradio界面
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()
theme.set(
    checkbox_label_background_fill_selected="*button_primary_background_fill",
    checkbox_label_text_color_selected="*button_primary_text_color",
)

with gr.Blocks(
        theme=theme,
        css="""
    .custom-log * {
        font-style: italic;
        font-size: 22px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        font-weight: bold !important;
        color: transparent !important;
        text-align: center !important;
    }

    .example-log * {
        font-style: italic;
        font-size: 16px !important;
        background-image: linear-gradient(120deg, #0ea5e9 0%, #6ee7b7 60%, #34d399 100%);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
    }

    #my_radio .wrap {
        display: flex;
        flex-wrap: nowrap;
        justify-content: center;
        align-items: center;
    }

    #my_radio .wrap label {
        display: flex;
        width: 50%;
        justify-content: center;
        align-items: center;
        margin: 0;
        padding: 10px 0;
        box-sizing: border-box;
    }
    """,
) as demo:
    # 使用隐藏的Textbox代替gr.State：
    is_example = gr.Textbox(label="is_example", visible=False, value="None")
    num_images = gr.Textbox(label="num_images", visible=False, value="None")

    gr.HTML(
        """
    <h1>🏛️ VGGT: 视觉几何基础Transformer</h1>
    <p>
    <a href="https://github.com/facebookresearch/vggt">🐙 GitHub 仓库</a> |
    <a href="#">项目主页</a>
    </p>

    <div style="font-size: 16px; line-height: 1.5;">
    <p>上传视频或一组图像来创建场景或物体的3D重建。VGGT接收这些图像并生成3D点云，以及估计的相机位姿。</p>

    <h3>开始使用：</h3>
    <ol>
        <li><strong>上传数据：</strong> 使用左侧的"上传视频"或"上传图像"按钮提供输入。视频将自动分割为单独的帧（每秒一帧）。</li>
        <li><strong>预览：</strong> 上传的图像将显示在左侧的画廊中。</li>
        <li><strong>重建：</strong> 点击"重建"按钮开始3D重建过程。</li>
        <li><strong>可视化：</strong> 3D重建将显示在右侧的查看器中。您可以旋转、平移和缩放来探索模型，并下载GLB文件。注意：对于大量输入图像，3D点的可视化可能会比较慢。</li>
        <li>
        <strong>调整可视化（可选）：</strong>
        重建后，您可以使用以下选项微调可视化效果
        <details style="display:inline;">
            <summary style="display:inline;">(<strong>点击展开</strong>)：</summary>
            <ul>
            <li><em>置信度阈值：</em> 基于置信度调整点的过滤。</li>
            <li><em>显示帧的点：</em> 选择要在点云中显示的特定帧。</li>
            <li><em>显示相机：</em> 切换估计相机位置的显示。</li>
            <li><em>过滤天空 / 过滤黑色背景：</em> 移除天空或黑色背景点。</li>
            <li><em>选择预测模式：</em> 在"深度图和相机分支"或"点图分支"之间选择。</li>
            </ul>
        </details>
        </li>
    </ol>
    <p><strong style="color: #0ea5e9;">请注意：</strong> <span style="color: #0ea5e9; font-weight: bold;">VGGT通常在1秒内完成场景重建。但是，由于第三方渲染，3D点的可视化可能需要数十秒，这与VGGT的处理时间无关。</span></p>
    </div>
    """
    )

    target_dir_output = gr.Textbox(label="目标目录", visible=False, value="None")

    with gr.Row():
        with gr.Column(scale=2):
            input_video = gr.Video(label="上传视频", interactive=True)
            input_images = gr.File(file_count="multiple", label="上传图像", interactive=True)

            image_gallery = gr.Gallery(
                label="预览",
                columns=4,
                height="300px",
                show_download_button=True,
                object_fit="contain",
                preview=True,
            )

        with gr.Column(scale=4):
            with gr.Column():
                gr.Markdown("**3D重建（点云和相机位姿）**")
                log_output = gr.Markdown(
                    "请上传视频或图像，然后点击重建。", elem_classes=["custom-log"]
                )
                reconstruction_output = gr.Model3D(height=520, zoom_speed=0.5, pan_speed=0.5)

            with gr.Row():
                submit_btn = gr.Button("重建", scale=1, variant="primary")
                clear_btn = gr.ClearButton(
                    [input_video, input_images, reconstruction_output, log_output, target_dir_output, image_gallery],
                    scale=1,
                    value="清除"
                )

            with gr.Row():
                prediction_mode = gr.Radio(
                    ["深度图和相机分支", "点图分支"],
                    label="选择预测模式",
                    value="深度图和相机分支",
                    scale=1,
                    elem_id="my_radio",
                )

            with gr.Row():
                conf_thres = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="置信度阈值 (%)")
                frame_filter = gr.Dropdown(choices=["全部"], value="全部", label="显示帧的点")
                with gr.Column():
                    show_cam = gr.Checkbox(label="显示相机", value=True)
                    mask_sky = gr.Checkbox(label="过滤天空", value=False)
                    mask_black_bg = gr.Checkbox(label="过滤黑色背景", value=False)
                    mask_white_bg = gr.Checkbox(label="过滤白色背景", value=False)

    # ---------------------- 示例部分 ----------------------
    examples = [
        [colosseum_video, "22", None, 20.0, False, False, True, False, "深度图和相机分支", "True"],
        [pyramid_video, "30", None, 35.0, False, False, True, False, "深度图和相机分支", "True"],
        [single_cartoon_video, "1", None, 15.0, False, False, True, False, "深度图和相机分支", "True"],
        [single_oil_painting_video, "1", None, 20.0, False, False, True, True, "深度图和相机分支", "True"],
        [room_video, "8", None, 5.0, False, False, True, False, "深度图和相机分支", "True"],
        [kitchen_video, "25", None, 50.0, False, False, True, False, "深度图和相机分支", "True"],
        [fern_video, "20", None, 45.0, False, False, True, False, "深度图和相机分支", "True"],
    ]


    def example_pipeline(
            input_video,
            num_images_str,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example_str,
    ):
        """
        1) 复制示例图像到新的target_dir
        2) 重建
        3) 返回model3D + logs + new_dir + 更新的下拉菜单 + 画廊
        我们不返回is_example。它只是一个输入。
        """
        target_dir, image_paths = handle_uploads(input_video, input_images)
        # 示例中总是使用"全部"作为frame_filter
        frame_filter = "全部"
        glbfile, log_msg, dropdown = gradio_demo(
            target_dir, conf_thres, frame_filter, mask_black_bg, mask_white_bg, show_cam, mask_sky, prediction_mode
        )
        return glbfile, log_msg, target_dir, dropdown, image_paths


    gr.Markdown("点击任何行来加载示例。", elem_classes=["example-log"])

    gr.Examples(
        examples=examples,
        inputs=[
            input_video,
            num_images,
            input_images,
            conf_thres,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        outputs=[reconstruction_output, log_output, target_dir_output, frame_filter, image_gallery],
        fn=example_pipeline,
        cache_examples=False,
        examples_per_page=50,
    )

    # -------------------------------------------------------------------------
    # "重建"按钮逻辑：
    #  - 清除字段
    #  - 更新日志
    #  - gradio_demo(...) 使用现有的target_dir
    #  - 然后设置is_example = "False"
    # -------------------------------------------------------------------------
    submit_btn.click(fn=clear_fields, inputs=[], outputs=[reconstruction_output]).then(
        fn=update_log, inputs=[], outputs=[log_output]
    ).then(
        fn=gradio_demo,
        inputs=[
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
        ],
        outputs=[reconstruction_output, log_output, frame_filter],
    ).then(
        fn=lambda: "False", inputs=[], outputs=[is_example]  # 设置is_example为"False"
    )

    # -------------------------------------------------------------------------
    # 实时可视化更新
    # -------------------------------------------------------------------------
    conf_thres.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    frame_filter.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_black_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_white_bg.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    show_cam.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    mask_sky.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )
    prediction_mode.change(
        update_visualization,
        [
            target_dir_output,
            conf_thres,
            frame_filter,
            mask_black_bg,
            mask_white_bg,
            show_cam,
            mask_sky,
            prediction_mode,
            is_example,
        ],
        [reconstruction_output, log_output],
    )

    # -------------------------------------------------------------------------
    # 当用户上传或更改文件时自动更新画廊
    # -------------------------------------------------------------------------
    input_video.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )
    input_images.change(
        fn=update_gallery_on_upload,
        inputs=[input_video, input_images],
        outputs=[reconstruction_output, target_dir_output, image_gallery, log_output],
    )

    demo.queue(max_size=20).launch(show_error=True, share=True)