#!/usr/bin/env python3
"""
HematoVision — Evaluation Script

Usage:
    python evaluate.py --model models/checkpoints/best_model.pth --data_dir data/raw/TEST
    python evaluate.py --model models/checkpoints/best_model.pth --data_dir data/raw/TEST --gradcam
"""

import argparse
import os

import torch

from src.models.model import HematoVisionClassifier
from src.data.dataset import BloodCellDataset, get_transforms
from src.utils.config import Config
from src.utils.logger import get_logger, print_banner
from src.evaluation.metrics import evaluate, print_evaluation_report, compute_class_accuracy
from src.evaluation.visualize import (
    plot_confusion_matrix,
    plot_per_class_metrics,
    plot_tsne,
)
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="HematoVision Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory with test images")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results/eval")
    parser.add_argument("--gradcam", action="store_true", help="Generate Grad-CAM visualizations")
    parser.add_argument("--tsne", action="store_true", help="Generate t-SNE embedding plot")
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()
    logger = get_logger("evaluate")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {args.model}")
    model = HematoVisionClassifier.from_checkpoint(args.model, device)
    model.to(device)

    dataset = BloodCellDataset(
        root=args.data_dir,
        transform=get_transforms(phase="test"),
    )
    logger.info(f"Dataset: {len(dataset)} images from {args.data_dir}")
    logger.info(f"Class distribution: {dataset.class_distribution()}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info("Running evaluation...")
    results = evaluate(model, loader, device, return_probs=True)
    print_evaluation_report(results)

    per_class_acc = compute_class_accuracy(results)
    logger.info("Per-class accuracy:")
    for cls, acc in per_class_acc.items():
        logger.info(f"  {cls}: {acc:.4f}")

    # Save plots
    os.makedirs(args.output_dir, exist_ok=True)

    plot_confusion_matrix(
        results["confusion_matrix"],
        save_path=os.path.join(args.output_dir, "confusion_matrix.png"),
    )
    plot_per_class_metrics(
        results["per_class"],
        save_path=os.path.join(args.output_dir, "per_class_metrics.png"),
    )
    logger.info(f"Plots saved → {args.output_dir}/")

    if args.tsne:
        logger.info("Extracting features for t-SNE...")
        all_features, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for imgs, lbls in loader:
                feats = model.get_features(imgs.to(device))
                if feats.dim() > 2:
                    feats = feats.mean(dim=[2, 3])
                all_features.append(feats.cpu().numpy())
                all_labels.extend(lbls.numpy().tolist())
        import numpy as np
        all_features = np.vstack(all_features)
        all_labels = np.array(all_labels)
        plot_tsne(all_features, all_labels, save_path=os.path.join(args.output_dir, "tsne.png"))
        logger.info("t-SNE plot saved")

    logger.info("Evaluation complete ✓")


if __name__ == "__main__":
    main()
