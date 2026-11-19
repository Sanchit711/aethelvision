"""Complete End-to-End Training & Validation Pipeline Demonstration Script.

Demonstrates the integrated lifecycle of Aurelix Detector:
Dataset -> Load Images -> Forward Pass -> Loss -> Backward -> Optimizer -> Checkpoint -> Validation -> mAP
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from datasets import CustomDataset
from aurelix_detector import AurelixDetector
from aurelix_detector.losses import CompositeDetectionLoss
from trainer import CheckpointManager, DetectorValidator, OptimizerFactory, TrainingConfig


def run_full_pipeline_demo() -> None:
    """Execute end-to-end dataset-to-validation pipeline lifecycle."""
    print("==================================================================")
    print("AURELIX DETECTOR - COMPLETE END-TO-END PIPELINE DEMONSTRATION")
    print("==================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Dataset Construction
    print("\n--- 1. DATASET INITIALIZATION ---")
    samples = [
        {
            "image": torch.randn(3, 320, 320),
            "boxes": [[20.0, 20.0, 100.0, 100.0]],
            "labels": [0.0],  # Person (Class 0)
        },
        {
            "image": torch.randn(3, 320, 320),
            "boxes": [[50.0, 50.0, 250.0, 250.0]],
            "labels": [2.0],  # Car (Class 2)
        },
    ]
    dataset = CustomDataset(samples=samples, img_size=(320, 320), max_gt=10)
    print(f"Dataset Size: {len(dataset)} samples")

    # 2. DataLoader Image Loading
    print("\n--- 2. LOADING IMAGES VIA DATALOADER ---")
    dataloader = DataLoader(dataset, batch_size=2)
    images, gt_labels, gt_bboxes, mask_gt = next(iter(dataloader))
    images, gt_labels, gt_bboxes, mask_gt = (
        images.to(device),
        gt_labels.to(device),
        gt_bboxes.to(device),
        mask_gt.to(device),
    )

    print(f"Loaded Images Batch Shape  : {list(images.shape)}")
    print(f"Loaded GT BBoxes Shape      : {list(gt_bboxes.shape)}")

    # 3. Model Forward Pass
    print("\n--- 3. MODEL FORWARD PASS ---")
    model = AurelixDetector(num_classes=80).to(device)
    model.train()
    raw_outputs = model(images)

    print(f"Raw Prediction Level 1 (N3): {list(raw_outputs[0].shape)}")
    print(f"Raw Prediction Level 2 (N4): {list(raw_outputs[1].shape)}")
    print(f"Raw Prediction Level 3 (N5): {list(raw_outputs[2].shape)}")

    # 4. Composite Loss Computation
    print("\n--- 4. LOSS COMPUTATION ---")
    loss_fn = CompositeDetectionLoss(num_classes=80).to(device)
    loss, loss_dict = loss_fn(raw_outputs, gt_labels, gt_bboxes, mask_gt)

    print(f"Total Composite Loss : {loss.item():.4f}")
    print(f"  Box IoU Loss (l_box): {loss_dict['l_box'].item():.4f}")
    print(f"  DFL Loss (l_dfl)    : {loss_dict['l_dfl'].item():.4f}")
    print(f"  Focal Loss (l_cls)  : {loss_dict['l_cls'].item():.4f}")

    # 5. Backward Pass
    print("\n--- 5. BACKWARD PASS ---")
    optimizer = OptimizerFactory.create_optimizer(model, optimizer_type="AdamW", learning_rate=1e-3)
    optimizer.zero_grad()
    loss.backward()
    print("Gradients computed successfully.")

    # 6. Optimizer Step
    print("\n--- 6. OPTIMIZER STEP ---")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    print("Optimizer step executed successfully.")

    # 7. Checkpointing
    print("\n--- 7. CHECKPOINT SAVE & LOAD ---")
    ckpt_mgr = CheckpointManager(checkpoint_dir="checkpoints_demo")
    ckpt_path = ckpt_mgr.save_checkpoint(
        model_state_dict=model.state_dict(),
        epoch=1,
        is_best=True,
    )
    print(f"Saved Checkpoint: {ckpt_path}")

    loaded_data = ckpt_mgr.load_checkpoint(ckpt_path)
    print("Loaded Checkpoint state dict successfully.")

    # 8. Validation & mAP Evaluation
    print("\n--- 8. VALIDATION & mAP EVALUATION ---")
    model.eval()
    validator = DetectorValidator(num_classes=80)

    # Decode predictions in eval mode
    with torch.no_grad():
        decoded_preds = model(images)

    pred_list = [decoded_preds[0], decoded_preds[1]]
    gt_boxes_list = [gt_bboxes[0][:1], gt_bboxes[1][:1]]
    gt_classes_list = [gt_labels[0][:1].squeeze(-1), gt_labels[1][:1].squeeze(-1)]

    metrics = validator.compute_metrics(pred_list, gt_boxes_list, gt_classes_list)

    print("\nFinal Validation Metrics:")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    print(f"  mAP@0.50   : {metrics['map_50']:.4f}")
    print(f"  mAP@0.50:95: {metrics['map_50_95']:.4f}")

    print("\n==================================================================")
    print("STATUS: Full Pipeline Lifecycle DEMO PASSED!")
    print("==================================================================\n")


if __name__ == "__main__":
    run_full_pipeline_demo()
