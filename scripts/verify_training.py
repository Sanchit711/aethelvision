"""Verification Script for Aurelix Detector Training Pipeline.

Demonstrates complete training iteration on synthetic data:
Synthetic Image -> Model Forward Pass -> Loss Computation -> Backward Pass -> Optimizer Step.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader, TensorDataset

from aurelix_detector import AurelixDetector
from aurelix_detector.losses import CompositeDetectionLoss
from trainer import DetectorTrainer, OptimizerFactory, TrainingConfig


def run_training_verification() -> None:
    """Demonstrate end-to-end training step on synthetic data."""
    print("==================================================================")
    print("AURELIX DETECTOR v0.3 - TRAINING PIPELINE VERIFICATION SUMMARY")
    print("==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Generate Synthetic Input Image Tensor (B=2, C=3, H=320, W=320)
    print("\n--- 1. GENERATING SYNTHETIC BATCH DATA ---")
    batch_size = 2
    images = torch.randn(batch_size, 3, 320, 320, device=device)
    gt_labels = torch.tensor([[[0.0]], [[2.0]]], device=device)
    gt_bboxes = torch.tensor([[[20.0, 20.0, 100.0, 100.0]], [[50.0, 50.0, 250.0, 250.0]]], device=device)
    mask_gt = torch.tensor([[[1.0]], [[1.0]]], device=device)

    print(f"Synthetic Images Tensor Shape : {list(images.shape)}")
    print(f"GT BBoxes Tensor Shape        : {list(gt_bboxes.shape)}")

    # 2. Model Forward Pass
    print("\n--- 2. EXECUTING MODEL FORWARD PASS ---")
    model = AurelixDetector(num_classes=80).to(device)
    model.train()
    raw_outputs = model(images)

    print(f"Raw Output Level 1 (N3 / Stride  8): {list(raw_outputs[0].shape)}")
    print(f"Raw Output Level 2 (N4 / Stride 16): {list(raw_outputs[1].shape)}")
    print(f"Raw Output Level 3 (N5 / Stride 32): {list(raw_outputs[2].shape)}")

    # 3. Loss Computation
    print("\n--- 3. COMPUTING COMPOSITE LOSS ---")
    loss_fn = CompositeDetectionLoss(num_classes=80).to(device)
    loss, loss_dict = loss_fn(raw_outputs, gt_labels, gt_bboxes, mask_gt)

    print(f"Total Composite Loss : {loss.item():.4f}")
    print(f"  Box IoU Loss (l_box): {loss_dict['l_box'].item():.4f}")
    print(f"  DFL Loss (l_dfl)    : {loss_dict['l_dfl'].item():.4f}")
    print(f"  Focal Loss (l_cls)  : {loss_dict['l_cls'].item():.4f}")

    # 4. Backward Pass
    print("\n--- 4. EXECUTING BACKWARD PASS ---")
    optimizer = OptimizerFactory.create_optimizer(model, optimizer_type="AdamW", learning_rate=1e-3)
    optimizer.zero_grad()
    loss.backward()
    print("Backward pass gradients computed successfully.")

    # 5. Optimizer Step
    print("\n--- 5. EXECUTING OPTIMIZER STEP ---")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    print("Optimizer step executed successfully.")

    print("\n==================================================================")
    print("STATUS: Image -> Forward -> Loss -> Backward -> Step VERIFIED!")
    print("==================================================================\n")


if __name__ == "__main__":
    run_training_verification()
