"""Validation & Inspection Script for End-to-End Aurelix Detector Pipeline.

Executes complete forward pass: Input (1, 3, 640, 640) -> Backbone -> Neck -> Detection Head -> Decoder.
Prints exact tensor shapes and model parameter metrics.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from aurelix_detector import AurelixDetector


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_full_pipeline_validation() -> None:
    """Execute end-to-end detector pipeline forward pass and print output tensor shapes."""
    print("==================================================================")
    print("AURELIX DETECTOR v0.2 - END-TO-END PIPELINE VALIDATION SUMMARY")
    print("==================================================================")

    x = torch.randn(1, 3, 640, 640)
    model = AurelixDetector(num_classes=80, depth_multiple=1.0, width_multiple=1.0)

    # 1. Training Mode Forward Pass (Raw Predictions)
    model.train()
    raw_preds = model(x)

    print("\n--- INPUT TENSOR ---")
    print(f"Input Image Tensor : {list(x.shape)}")

    print("\n--- RAW DETECTION HEAD OUTPUTS (TRAINING MODE) ---")
    print(f"Scale 1 (N3 / Stride  8): {list(raw_preds[0].shape)} (80 classes + 64 DFL reg)")
    print(f"Scale 2 (N4 / Stride 16): {list(raw_preds[1].shape)} (80 classes + 64 DFL reg)")
    print(f"Scale 3 (N5 / Stride 32): {list(raw_preds[2].shape)} (80 classes + 64 DFL reg)")

    # 2. Eval Mode Forward Pass (Decoded Bounding Boxes & Class Probabilities)
    model.eval()
    with torch.no_grad():
        decoded_preds = model(x)

    print("\n--- DECODED PREDICTIONS (EVAL MODE) ---")
    print(f"Decoded Tensor Shape    : {list(decoded_preds.shape)} (B, Total_Anchors, 4_coords + 80_classes)")

    # 3. Model Parameters
    params = count_parameters(model)
    print("\n--- MODEL METRICS ---")
    print(f"Total Detector Parameters: {params:,} parameters ({params / 1e6:.2f} M)")
    print("STATUS: End-to-End Forward Pass Validation PASSED!\n")


if __name__ == "__main__":
    run_full_pipeline_validation()
