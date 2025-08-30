import logging
from pathlib import Path
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from omegaconf import DictConfig
from hydra.utils import instantiate
import time
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    def __init__(self, **kwargs):
        self.cfg = DictConfig(kwargs) if not isinstance(kwargs, DictConfig) else kwargs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.epoch = 0
        self.step = 0

        self._setup_logging()
        self._setup_model()
        self._setup_training()
        self._setup_data()

        self.use_amp = self.cfg.optim.amp.enabled and self.device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)

    def _setup_logging(self):
        self.log_dir = Path(self.cfg.logging.log_dir) / self.cfg.exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'train.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _setup_model(self):
        self.model = instantiate(self.cfg.model).to(self.device)

        if self.cfg.checkpoint.resume_checkpoint_path:
            self.load_checkpoint(self.cfg.checkpoint.resume_checkpoint_path)

    def _setup_training(self):
        # Freeze modules
        frozen = self.cfg.optim.get('frozen_module_names', [])
        for pattern in frozen:
            if hasattr(self.model, pattern):
                module = getattr(self.model, pattern)
                if module:
                    for p in module.parameters():
                        p.requires_grad = False

        # Setup optimizer
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = instantiate(self.cfg.optim.optimizer, params=params)

        # Setup loss
        from loss import MultitaskLoss
        self.loss_fn = MultitaskLoss(**self.cfg.loss)

        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.max_epochs,
            eta_min=1e-7
        ) if self.cfg.optim.get('scheduler') else None

        self.clip_grad = self.cfg.optim.gradient_clip.get('max_norm')

    def _setup_data(self):
        data = self.cfg.data

        if 'train' in data:
            self.train_loader = instantiate(data.train)
        else:
            self.train_loader = None

        if 'val' in data:
            self.val_loader = instantiate(data.val)
        else:
            self.val_loader = None

    def run(self):
        for epoch in range(self.cfg.max_epochs):
            self.epoch = epoch
            self.logger.info(f"\nEpoch {epoch + 1}/{self.cfg.max_epochs}")

            if self.train_loader:
                metrics = self.train_epoch()
                self.log_metrics("Train", metrics)

            if self.val_loader and (epoch + 1) % self.cfg.val_epoch_freq == 0:
                metrics = self.validate()
                self.log_metrics("Val", metrics)

            if (epoch + 1) % self.cfg.checkpoint.save_freq == 0:
                self.save_checkpoint(epoch)

            if self.scheduler:
                self.scheduler.step()

    # 在Trainer类的train_epoch方法中修改：

    def train_epoch(self):
        self.model.train()
        tracker = MetricTracker()

        train_loader = self.train_loader.get_loader(self.epoch)

        for i, batch in enumerate(train_loader):
            batch = self.prepare_batch(batch)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                pred = self.model(batch['images'], epoch=self.epoch)
                losses = self.loss_fn(pred, batch, epoch=self.epoch)

            self.backward_step(losses['loss_objective'])
            tracker.update(losses)
            self.step += 1

            if i % self.cfg.logging.log_freq == 0:
                pos_samples = 0
                if 'pred_scores' in pred:
                    pos_samples = (pred['pred_scores'] > 0.5).sum().item()

                log_msg = f"Step {self.step}: {self.format_losses(losses)}"
                log_msg += f" | Pos samples: {pos_samples}"
                self.logger.info(log_msg)

        return tracker.average()

    def validate(self):
        self.model.eval()
        tracker = MetricTracker()
        val_loader = self.val_loader.get_loader(self.epoch)

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                batch = self.prepare_batch(batch)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    pred = self.model(batch['images'])
                    losses = self.loss_fn(pred, batch)

                tracker.update(losses)

        return tracker.average()

    def backward_step(self, loss):
        self.optimizer.zero_grad(set_to_none=True)

        if self.use_amp:
            self.scaler.scale(loss).backward()
            if self.clip_grad:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

    def prepare_batch(self, batch):
        if self.step == 0:
            print(f"[DEBUG Trainer] prepare_batch input:")
            if 'gt_boxes' in batch:
                print(f"  gt_boxes type: {type(batch['gt_boxes'])}")
                if isinstance(batch['gt_boxes'], list):
                    print(f"  gt_boxes list length: {len(batch['gt_boxes'])}")

        if 'gt_boxes' in batch:
            if isinstance(batch['gt_boxes'], list):
                batch['gt_boxes'] = [box.to(self.device) for box in batch['gt_boxes']]
            else:
                raise ValueError(f"gt_boxes should be list, got {type(batch['gt_boxes'])}")

        if 'gt_classes' in batch:
            if isinstance(batch['gt_classes'], list):
                batch['gt_classes'] = [cls.to(self.device) for cls in batch['gt_classes']]
            else:
                raise ValueError(f"gt_classes should be list, got {type(batch['gt_classes'])}")

        # 处理标准张量
        for k in ['images', 'depths', 'extrinsics', 'intrinsics',
                  'cam_points', 'world_points', 'point_masks', 'valid_frames_mask']:
            if k in batch and torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(self.device, non_blocking=True)

        # 处理可能是列表的数据
        if 'ids' in batch and isinstance(batch['ids'], list):
            batch['ids'] = [
                id_tensor.to(self.device) if torch.is_tensor(id_tensor) else torch.tensor(id_tensor).to(self.device)
                for id_tensor in batch['ids']]
        elif 'ids' in batch and torch.is_tensor(batch['ids']):
            batch['ids'] = batch['ids'].to(self.device)

        return batch

    def _normalize_gt_data(self, gt_data, expected_batch_size, data_type):
        """标准化GT数据格式，确保长度与batch size匹配"""
        if gt_data is None:
            gt_data = []

        # 如果是张量，转换为列表
        if torch.is_tensor(gt_data):
            if gt_data.ndim == 3 and gt_data.shape[0] == expected_batch_size:
                # [B, N, D] 格式
                gt_data = [gt_data[i] for i in range(expected_batch_size)]
            else:
                # 单个张量，复制到所有batch
                gt_data = [gt_data for _ in range(expected_batch_size)]

        # 如果不是列表，转换为列表
        if not isinstance(gt_data, list):
            gt_data = [gt_data] * expected_batch_size

        # 确保列表长度与batch size匹配
        if len(gt_data) < expected_batch_size:
            # 补齐数据
            for _ in range(expected_batch_size - len(gt_data)):
                if data_type == 'boxes':
                    gt_data.append(torch.zeros((0, 7), dtype=torch.float32, device=self.device))
                else:  # classes
                    gt_data.append(torch.zeros((0,), dtype=torch.long, device=self.device))
        elif len(gt_data) > expected_batch_size:
            # 截断数据
            gt_data = gt_data[:expected_batch_size]

        # 确保所有元素都是张量并在正确设备上
        for i in range(len(gt_data)):
            if not torch.is_tensor(gt_data[i]):
                if data_type == 'boxes':
                    if gt_data[i] is None or (hasattr(gt_data[i], '__len__') and len(gt_data[i]) == 0):
                        gt_data[i] = torch.zeros((0, 7), dtype=torch.float32)
                    else:
                        gt_data[i] = torch.tensor(gt_data[i], dtype=torch.float32)
                else:  # classes
                    if gt_data[i] is None or (hasattr(gt_data[i], '__len__') and len(gt_data[i]) == 0):
                        gt_data[i] = torch.zeros((0,), dtype=torch.long)
                    else:
                        gt_data[i] = torch.tensor(gt_data[i], dtype=torch.long)

            # 移动到设备
            gt_data[i] = gt_data[i].to(self.device)

        return gt_data

    def collate(self, batch_list):
        """自定义collate函数，正确处理GT数据"""
        if not batch_list:
            return {}

        result = {}
        batch_size = len(batch_list)

        for key in batch_list[0].keys():
            if key == 'seq_name':
                result[key] = [item[key] for item in batch_list]
            elif key in ['gt_boxes', 'gt_classes']:
                # 收集所有GT数据
                all_gt_data = []
                for item in batch_list:
                    if key in item and item[key] is not None:
                        gt_data = item[key]
                        if torch.is_tensor(gt_data):
                            all_gt_data.append(gt_data)
                        elif isinstance(gt_data, (list, tuple)):
                            # 如果是嵌套列表，展平
                            for sub_item in gt_data:
                                all_gt_data.append(sub_item)
                        else:
                            all_gt_data.append(gt_data)
                    else:
                        # 添加空数据
                        if key == 'gt_boxes':
                            all_gt_data.append(torch.zeros((0, 7), dtype=torch.float32))
                        else:  # gt_classes
                            all_gt_data.append(torch.zeros((0,), dtype=torch.long))

                result[key] = all_gt_data
            else:
                try:
                    if key in batch_list[0] and torch.is_tensor(batch_list[0][key]):
                        result[key] = torch.stack([item[key] for item in batch_list])
                    else:
                        result[key] = [item[key] for item in batch_list]
                except Exception as e:
                    # 如果堆叠失败，保持列表格式
                    result[key] = [item.get(key) for item in batch_list]

        return result

    def save_checkpoint(self, epoch):
        ckpt_dir = Path(self.cfg.checkpoint.save_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'step': self.step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict() if self.use_amp else None,
            'config': self.cfg
        }

        path = ckpt_dir / f'epoch_{epoch:04d}.pth'
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)

        if 'optimizer' in checkpoint and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        if 'scaler' in checkpoint and self.use_amp:
            self.scaler.load_state_dict(checkpoint['scaler'])

        self.epoch = checkpoint.get('epoch', 0)
        self.step = checkpoint.get('step', 0)

    def format_losses(self, losses):
        items = []
        for k, v in losses.items():
            if torch.is_tensor(v):
                items.append(f"{k}={v.item():.4f}")
        return ", ".join(items)

    def log_metrics(self, phase, metrics):
        if not metrics:
            return

        self.logger.info(f"{phase} Metrics: {self.format_losses(metrics)}")


class MetricTracker:
    def __init__(self):
        self.values = {}
        self.counts = {}

    def update(self, metrics):
        for k, v in metrics.items():
            if torch.is_tensor(v):
                v = v.item() if v.numel() == 1 else None

            if v is not None and isinstance(v, (int, float)):
                if k not in self.values:
                    self.values[k] = 0.0
                    self.counts[k] = 0
                self.values[k] += v
                self.counts[k] += 1

    def average(self):
        return {k: v / self.counts[k] for k, v in self.values.items() if self.counts[k] > 0}