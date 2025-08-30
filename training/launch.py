#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path
from hydra import initialize, compose
from omegaconf import OmegaConf
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--config', '-c', type=str, default='detection')
    parser.add_argument('--override', '-o', nargs='*', default=[])
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")

    with initialize(version_base=None, config_path="config"):
        cfg = compose(config_name=args.config)

        overrides = args.override
        if args.debug:
            overrides.extend([
                'max_epochs=2',
                'limit_train_batches=5',
                'limit_val_batches=2'
            ])

        for override in overrides:
            key, value = override.split('=')
            OmegaConf.update(cfg, key, value, merge=False)

    print(f"\nExperiment: {cfg.exp_name}")
    print(f"Model: {cfg.model._target_.split('.')[-1]}")
    print(f"Epochs: {cfg.max_epochs}")
    print(f"Learning rate: {cfg.optim.optimizer.lr}")

    try:
        from trainer import Trainer
        trainer = Trainer(**cfg)
        trainer.run()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()