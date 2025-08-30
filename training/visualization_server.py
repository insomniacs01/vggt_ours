# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import json
import torch
import trimesh
import numpy as np
from flask import Flask, send_file, jsonify, render_template
from pathlib import Path
import threading

app = Flask(__name__)

vis_state = {
    'glb_path': None,
    'lock': threading.Lock(),
    'directory': Path('logs/visuals')
}


def set_vis_directory(path):
    vis_state['directory'] = Path(path)
    vis_state['directory'].mkdir(parents=True, exist_ok=True)


@app.route('/')
def index():
    """主页面"""
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>3D Training Monitor</title>
        <style>
            body { margin: 0; background: #1a1a2e; color: white; font-family: monospace; }
            #info { position: absolute; top: 20px; left: 20px; background: rgba(0,0,0,0.8); padding: 15px; }
            #viewer { width: 100vw; height: 100vh; }
        </style>
    </head>
    <body>
        <div id="info">
            <h3>3D Detection Monitor</h3>
            <p id="status">Loading...</p>
        </div>
        <div id="viewer"></div>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/loaders/GLTFLoader.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
        <script>
            const scene = new THREE.Scene();
            scene.background = new THREE.Color(0x1a1a2e);

            const camera = new THREE.PerspectiveCamera(75, window.innerWidth/window.innerHeight, 0.1, 1000);
            camera.position.set(15, 15, 15);

            const renderer = new THREE.WebGLRenderer({antialias: true});
            renderer.setSize(window.innerWidth, window.innerHeight);
            document.getElementById('viewer').appendChild(renderer.domElement);

            scene.add(new THREE.AmbientLight(0xffffff, 0.5));
            const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
            dirLight.position.set(10, 20, 10);
            scene.add(dirLight);

            const controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;

            const loader = new THREE.GLTFLoader();
            let currentModel = null;

            function loadScene() {
                fetch('/api/scene')
                    .then(r => r.blob())
                    .then(blob => {
                        const url = URL.createObjectURL(blob);
                        loader.load(url, (gltf) => {
                            if (currentModel) scene.remove(currentModel);
                            currentModel = gltf.scene;
                            scene.add(currentModel);
                            document.getElementById('status').textContent = 'Updated: ' + new Date().toLocaleTimeString();
                        });
                    })
                    .catch(err => {
                        document.getElementById('status').textContent = 'Waiting for data...';
                    });
            }

            function animate() {
                requestAnimationFrame(animate);
                controls.update();
                renderer.render(scene, camera);
            }

            loadScene();
            setInterval(loadScene, 5000);
            animate();

            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });
        </script>
    </body>
    </html>
    '''


@app.route('/api/scene')
def get_scene():
    """返回最新的3D场景"""
    glb_path = vis_state['directory'] / 'latest_scene.glb'

    if glb_path.exists():
        return send_file(str(glb_path), mimetype='model/gltf-binary')

    empty = create_empty_scene()
    temp_path = vis_state['directory'] / 'empty.glb'
    empty.export(str(temp_path))
    return send_file(str(temp_path), mimetype='model/gltf-binary')


@app.route('/api/status')
def get_status():
    """返回状态信息"""
    glb_path = vis_state['directory'] / 'latest_scene.glb'
    return jsonify({
        'has_data': glb_path.exists(),
        'last_modified': glb_path.stat().st_mtime if glb_path.exists() else None
    })


def create_empty_scene():
    """创建空场景"""
    scene = trimesh.Scene()

    ground = trimesh.creation.box(extents=[20, 0.1, 20])
    ground.visual.face_colors = [100, 100, 100, 100]
    scene.add_geometry(ground)

    for axis, color in zip([[5, 0, 0], [0, 5, 0], [0, 0, 5]],
                           [[255, 0, 0], [0, 255, 0], [0, 0, 255]]):
        arrow = trimesh.creation.cylinder(radius=0.1, height=5)
        arrow.visual.face_colors = color + [255]
        transform = np.eye(4)
        transform[:3, 3] = np.array(axis) / 2
        arrow.apply_transform(transform)
        scene.add_geometry(arrow)

    return scene


def predictions_to_glb(predictions, gt_boxes=None, save_path=None):
    """转换预测结果为GLB格式"""
    scene = trimesh.Scene()

    ground = trimesh.creation.box(extents=[30, 0.1, 30])
    ground.visual.face_colors = [50, 50, 50, 100]
    scene.add_geometry(ground)

    if 'pred_boxes' in predictions:
        _add_boxes(scene, predictions['pred_boxes'], [255, 50, 50, 150], 'pred')

    if gt_boxes is not None:
        _add_boxes(scene, gt_boxes, [50, 255, 50, 150], 'gt')

    if 'world_points' in predictions:
        _add_points(scene, predictions['world_points'])

    if save_path:
        scene.export(save_path)
        with vis_state['lock']:
            vis_state['glb_path'] = save_path

    return scene


def _add_boxes(scene, boxes, color, prefix):
    """添加3D框到场景"""
    if torch.is_tensor(boxes):
        boxes = boxes.cpu().numpy()

    if len(boxes.shape) == 4:
        boxes = boxes[0, 0]
    elif len(boxes.shape) == 3:
        boxes = boxes[0]

    for i, box in enumerate(boxes[:20]):
        if len(box) < 6:
            continue

        mesh = _create_box_mesh(
            center=box[:3],
            size=box[3:6],
            yaw=box[6] if len(box) > 6 else 0
        )
        mesh.visual.face_colors = color
        scene.add_geometry(mesh, node_name=f'{prefix}_box_{i}')


def _add_points(scene, points, max_points=5000):
    """添加点云到场景"""
    if torch.is_tensor(points):
        points = points.cpu().numpy()

    points = points.reshape(-1, 3)

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]

    valid = np.all(np.abs(points) < 50, axis=1)
    points = points[valid]

    if len(points) > 0:
        colors = np.tile([255, 255, 100, 255], (len(points), 1))
        cloud = trimesh.PointCloud(points, colors=colors)
        scene.add_geometry(cloud, node_name='points')


def _create_box_mesh(center, size, yaw=0):
    """创建3D框mesh"""
    size = np.maximum(np.abs(size), 0.1)

    box = trimesh.creation.box(extents=size)

    transform = trimesh.transformations.rotation_matrix(yaw, [0, 1, 0])
    transform[:3, 3] = center
    box.apply_transform(transform)

    return box


class VisualizationHelper:
    """可视化助手类"""

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        set_vis_directory(str(self.output_dir))

    def update(self, predictions, batch=None):
        save_path = self.output_dir / 'latest_scene.glb'

        gt_boxes = None
        if batch and 'gt_boxes' in batch:
            gt_boxes = batch['gt_boxes']

        predictions_to_glb(predictions, gt_boxes, str(save_path))

    def start_server(self, port=5000):
        import threading
        thread = threading.Thread(
            target=lambda: app.run(host='0.0.0.0', port=port, debug=False)
        )
        thread.daemon = True
        thread.start()
        return thread