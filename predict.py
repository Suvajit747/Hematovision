#!/usr/bin/env python3
"""
HematoVision — Single Image Prediction

Usage:
    python predict.py --image path/to/cell.jpg --model models/checkpoints/best_model.pth
    python predict.py --image path/to/cell.jpg --model best_model.pth --gradcam
"""

import argparse
import sys

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from src.models.model import HematoVisionClassifier
from src.data.dataset import get_transforms, LABEL_NAMES, IDX_TO_CLASS
from src.utils.logger import get_logger, print_banner


CELL_INFO = {
    "EOSINOPHIL": {
        "normal_range": "1–4%",
        "function": "Combats parasites and mediates allergic responses",
        "morphology": "Bilobed nucleus, bright red/orange granules",
        "clinical": "Elevated in allergies, asthma, parasitic infections",
    },
    "LYMPHOCYTE": {
        "normal_range": "20–40%",
        "function": "Adaptive immunity (B cells → antibodies, T cells → cellular immunity)",
        "morphology": "Large round nucleus, scant blue cytoplasm",
        "clinical": "Elevated in viral infections, lymphocytic leukemia",
    },
    "MONOCYTE": {
        "normal_range": "2–8%",
        "function": "Phagocytosis; differentiates into macrophages/dendritic cells",
        "morphology": "Kidney/horseshoe-shaped nucleus, abundant gray cytoplasm",
        "clinical": "Elevated in chronic inflammation, monocytic leukemia",
    },
    "NEUTROPHIL": {
        "normal_range": "55–70%",
        "function": "First responder to bacterial infection; phagocytosis",
        "morphology": "Multi-lobed (3–5) nucleus, fine pink granules",
        "clinical": "Elevated in bacterial infections, inflammatory conditions",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="HematoVision Single-Image Prediction")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--gradcam", action="store_true")
    parser.add_argument("--top_k", type=int, default=4, help="Show top-k class probabilities")
    return parser.parse_args()


def predict(image_path: str, model, device: str, top_k: int = 4):
    transform = get_transforms(phase="test")
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    top_indices = np.argsort(probs)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "class": LABEL_NAMES[idx],
            "class_key": IDX_TO_CLASS.get(idx, LABEL_NAMES[idx]).upper(),
            "probability": float(probs[idx]),
            "rank": len(results) + 1,
        })

    return results, tensor.squeeze(0)


def main():
    print_banner()
    args = parse_args()
    logger = get_logger("predict")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading model from {args.model}")
    model = HematoVisionClassifier.from_checkpoint(args.model, device)

    logger.info(f"Predicting: {args.image}")
    predictions, tensor = predict(args.image, model, device, args.top_k)

    # Print results
    print("\n" + "═" * 50)
    print("  HEMATOVISION PREDICTION REPORT")
    print("═" * 50)
    top = predictions[0]
    print(f"\n  🔬 Predicted Class: {top['class'].upper()}")
    print(f"  📊 Confidence:      {top['probability']*100:.1f}%")

    info = CELL_INFO.get(top["class_key"].upper(), {})
    if info:
        print(f"\n  Normal Range:  {info['normal_range']}")
        print(f"  Function:      {info['function']}")
        print(f"  Morphology:    {info['morphology']}")
        print(f"  Clinical note: {info['clinical']}")

    print("\n  All class probabilities:")
    for p in predictions:
        bar = "█" * int(p["probability"] * 30)
        print(f"    {p['class']:<14} {bar:<30} {p['probability']*100:5.1f}%")
    print("═" * 50 + "\n")

    if args.gradcam:
        from src.evaluation.visualize import visualize_gradcam
        logger.info("Generating Grad-CAM...")
        fig = visualize_gradcam(model, tensor, save_path="results/gradcam.png")
        if fig:
            logger.info("Grad-CAM saved → results/gradcam.png")

    return predictions


if __name__ == "__main__":
    main()
