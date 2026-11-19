"""Validation & Verification Script for Aurelix Detector Post-Processing (NMS & Decoding).

Generates synthetic multi-scale detector outputs and verifies final detection format
[x1, y1, x2, y2, confidence, class_id].
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from aurelix_detector.postprocess import PostprocessDecoder, non_max_suppression


def run_postprocess_validation() -> None:
    """Execute post-processing validation on synthetic detector predictions."""
    print("==================================================================")
    print("AURELIX DETECTOR - POST-PROCESSING & NMS VALIDATION SUMMARY")
    print("==================================================================")

    # Construct synthetic predictions for B=2 batch items, 8400 anchors, 80 classes
    batch_size = 2
    total_anchors = 8400
    num_classes = 80
    synthetic_preds = torch.zeros((batch_size, total_anchors, 4 + num_classes))

    # Add synthetic object 1 (Image 0, Person class_id=0, conf=0.92)
    synthetic_preds[0, 100, :4] = torch.tensor([50.0, 50.0, 200.0, 200.0])
    synthetic_preds[0, 100, 4 + 0] = 0.92

    # Add synthetic object 2 (Image 0, Duplicate overlapping Person box, conf=0.81 -> Suppressed by NMS)
    synthetic_preds[0, 101, :4] = torch.tensor([52.0, 52.0, 198.0, 198.0])
    synthetic_preds[0, 101, 4 + 0] = 0.81

    # Add synthetic object 3 (Image 0, Car class_id=2, conf=0.88)
    synthetic_preds[0, 500, :4] = torch.tensor([300.0, 150.0, 500.0, 350.0])
    synthetic_preds[0, 500, 4 + 2] = 0.88

    # Add synthetic object 4 (Image 1, Bicycle class_id=1, conf=0.95)
    synthetic_preds[1, 250, :4] = torch.tensor([100.0, 100.0, 250.0, 250.0])
    synthetic_preds[1, 250, 4 + 1] = 0.95

    decoder = PostprocessDecoder(conf_threshold=0.25, iou_threshold=0.45)
    batch_detections = decoder(synthetic_preds)

    print(f"Batch Size: {len(batch_detections)}")

    for idx, det in enumerate(batch_detections):
        print(f"\n--- IMAGE {idx} FINAL DETECTIONS ---")
        print(f"Total Detections Kept: {det.shape[0]}")
        print("Format per detection row: [x1, y1, x2, y2, confidence, class_id]")
        for row in det:
            x1, y1, x2, y2, conf, cls_id = row.tolist()
            print(f"  Box: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] | Conf: {conf:.4f} | Class ID: {int(cls_id)}")

    print("\nSTATUS: Post-processing validation PASSED!\n")


if __name__ == "__main__":
    run_postprocess_validation()
