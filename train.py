#!/usr/bin/env python3
"""
HematoVision — Main Training Script

Usage:
    python train.py
    python train.py --backbone resnet50 --epochs 40 --batch_size 32
    python train.py --backbone efficientnet_b3 --lr 5e-5 --use_amp
    python train.py --config configs/experiment.yaml
    python train.py --resume models/checkpoints/epoch_010.pth
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

from src.utils.config import Config
from src.utils.logger import get_logger, print_banner, print_config
from src.data.dataset import build_dataloaders
from src.models.model import build_model
from src.training.trainer import HematoTrainer
from src.evaluation.metrics import evaluate, print_evaluation_report
from src.evaluation.visualize import (
    plot_confusion_matrix,
    plot_training_history,
    plot_per_class_metrics,
)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="HematoVision Training")
    parser.add_argument("--config", type=str, default=None, help="YAML config path")
    parser.add_argument("--backbone", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default=None, help="Path to TRAIN/ folder")
    parser.add_argument("--val_dir", type=str, default=None, help="Path to VAL/ folder (optional)")
    parser.add_argument("--test_dir", type=str, default=None, help="Path to TEST/ folder (optional)")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--experiment", type=str, default="hematovision_run")
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()

    # ── Load config ────────────────────────────────────────────────────────────
    cfg = Config.load(args.config) if args.config else Config()

    # CLI overrides
    if args.backbone:     cfg.model.backbone = args.backbone
    if args.epochs:       cfg.training.epochs = args.epochs
    if args.batch_size:   cfg.training.batch_size = args.batch_size
    if args.lr:           cfg.training.learning_rate = args.lr
    if args.data_dir:     cfg.data.data_dir = args.data_dir
    if args.freeze_backbone: cfg.model.freeze_backbone = True
    if args.use_amp:      cfg.training.use_amp = True
    cfg.experiment_name = args.experiment

    logger = get_logger("train", cfg.log_dir)
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    print_config(cfg)
    cfg.save(os.path.join(cfg.training.checkpoint_dir, "config.yaml"))

    # ── Resolve data dirs ──────────────────────────────────────────────────────
    train_dir = args.data_dir or os.path.join(cfg.data.data_dir, "TRAIN")
    val_dir   = args.val_dir  or os.path.join(cfg.data.data_dir, "VAL")
    test_dir  = args.test_dir or os.path.join(cfg.data.data_dir, "TEST")

    # Fallback: use TEST as val if no VAL split exists
    if not os.path.exists(val_dir) and os.path.exists(test_dir):
        val_dir = test_dir

    logger.info(f"Train: {train_dir}")
    logger.info(f"Val:   {val_dir}")
    logger.info(f"Test:  {test_dir if os.path.exists(test_dir) else 'N/A'}")

    # ── Build dataloaders ──────────────────────────────────────────────────────
    dataloaders = build_dataloaders(
        cfg,
        train_dir=train_dir if os.path.exists(train_dir) else None,
        val_dir=val_dir if os.path.exists(val_dir) else None,
        test_dir=test_dir if os.path.exists(test_dir) else None,
    )

    if "train" not in dataloaders:
        logger.error("No training data found. Set --data_dir or check data/ folder structure.")
        sys.exit(1)

    # ── Build model ────────────────────────────────────────────────────────────
    if args.resume:
        from src.models.model import HematoVisionClassifier
        model = HematoVisionClassifier.from_checkpoint(args.resume, device)
        logger.info(f"Resumed from {args.resume}")
    else:
        model = build_model(cfg)

    params = model.count_parameters()
    logger.info(
        f"Model: {cfg.model.backbone} | "
        f"Total: {params['total']:,} | "
        f"Trainable: {params['trainable']:,}"
    )

    # ── Train ──────────────────────────────────────────────────────────────────
    trainer = HematoTrainer(model, dataloaders, cfg, device)
    history = trainer.train()

    # ── Save plots ─────────────────────────────────────────────────────────────
    os.makedirs(cfg.results_dir, exist_ok=True)
    fig = plot_training_history(history, save_path=os.path.join(cfg.results_dir, "training_history.png"))
    logger.info(f"Training history saved → {cfg.results_dir}/training_history.png")

    # ── Final evaluation ───────────────────────────────────────────────────────
    if "test" in dataloaders or "val" in dataloaders:
        eval_loader = dataloaders.get("test", dataloaders.get("val"))
        logger.info("Running final evaluation...")

        # Load best model for evaluation
        best_path = os.path.join(cfg.training.checkpoint_dir, "best_model.pth")
        if os.path.exists(best_path):
            from src.models.model import HematoVisionClassifier
            model = HematoVisionClassifier.from_checkpoint(best_path, device)

        results = evaluate(model, eval_loader, device)
        print_evaluation_report(results)

        plot_confusion_matrix(
            results["confusion_matrix"],
            save_path=os.path.join(cfg.results_dir, "confusion_matrix.png"),
        )
        plot_per_class_metrics(
            results["per_class"],
            save_path=os.path.join(cfg.results_dir, "per_class_metrics.png"),
        )
        logger.info(f"Evaluation plots saved → {cfg.results_dir}/")

    logger.info("Done! ✓")


if __name__ == "__main__":
    main()
