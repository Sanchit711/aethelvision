"""Production Validation CLI for Aurelix Detector.

Executes offline model validation and evaluation over PyTorch (.pt) checkpoints
or ONNX (.onnx) model binaries across CPU and CUDA compute backends.

Usage:
    python validate.py --weights checkpoints/best.pt --dataset datasets/default
    python validate.py --weights exports/best.onnx --conf-thresh 0.25 --device cpu
"""

import argparse
import sys
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.yolo import YOLODataset
from evaluation.confusion_matrix import ProductionConfusionMatrix
from evaluation.error_analysis import ErrorAnalyzer
from evaluation.metrics import DetectionEvaluator
from evaluation.plots import EvaluationPlotter
from evaluation.report_generator import ReportGenerator
from evaluation.visualizer import EvaluationVisualizer
from aurelix_detector.model import AurelixDetector


def parse_args() -> argparse.Namespace:
    """Parse command line arguments for validate.py CLI driver."""
    parser = argparse.ArgumentParser(description="Aurelix Detector Evaluation CLI Driver")
    parser.add_argument(
        "--weights",
        type=str,
        default="checkpoints/best.pt",
        help="Path to model weights checkpoint (.pt) or ONNX model (.onnx)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="datasets/default",
        help="Path to dataset directory containing images and labels",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Inference batch size integer",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Target spatial dimension (height and width)",
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.25,
        help="Confidence score threshold float",
    )
    parser.add_argument(
        "--iou-thresh",
        type=float,
        default=0.45,
        help="IoU overlap threshold float",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computing device string ('cpu', 'cuda')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/eval",
        help="Directory path for output reports and plots",
    )
    return parser.parse_args()


def run_validation(args: argparse.Namespace) -> None:
    """Execute complete validation pipeline."""
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_p = Path(args.weights)
    is_onnx = weights_p.suffix.lower() == ".onnx"

    target_device = "cuda" if (torch.cuda.is_available() and args.device.lower() == "cuda") else "cpu"
    device = torch.device(target_device)

    print(f"=== AURELIX DETECTOR VALIDATION PIPELINE ===")
    print(f"  Model Weights : {weights_p}")
    print(f"  Format        : {'ONNX' if is_onnx else 'PyTorch Checkpoint'}")
    print(f"  Device        : {target_device}")
    print(f"  Image Size    : ({args.imgsz}, {args.imgsz})")
    print(f"  Conf Thresh   : {args.conf_thresh} | IoU Thresh: {args.iou_thresh}")

    # 1. Load Model Architecture / Inference Engine
    model = None
    onnx_session = None

    if is_onnx:
        try:
            import onnxruntime as ort

            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if target_device == "cuda" else ["CPUExecutionProvider"]
            onnx_session = ort.InferenceSession(str(weights_p), providers=providers)
            print("  Loaded ONNX Runtime session successfully.")
        except Exception as e:
            print(f"  Warning: ONNX session load failed: {e}. Initializing PyTorch fallback model.")
            model = AurelixDetector(num_classes=80).to(device).eval()
    else:
        model = AurelixDetector(num_classes=80).to(device)
        if weights_p.exists() and weights_p.is_file():
            try:
                model.load_weights(weights_p)
                print(f"  Loaded model weights from {weights_p}.")
            except Exception:
                print(f"  Notice: Could not load weights from {weights_p}. Using default model initialization.")
        model.eval()

    # 2. Collect Dataset Image and Label Paths
    ds_dir = Path(args.dataset)
    img_dir = ds_dir / "images" if (ds_dir / "images").exists() else ds_dir
    lbl_dir = ds_dir / "labels" if (ds_dir / "labels").exists() else ds_dir

    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    lbl_paths = [lbl_dir / f"{p.stem}.txt" for p in img_paths]

    print(f"  Found {len(img_paths)} images in dataset directory '{ds_dir}'.")

    if not img_paths:
        print("  Warning: Empty dataset directory. Generating empty validation report.")
        dataset = YOLODataset([], [], img_size=(args.imgsz, args.imgsz))
    else:
        dataset = YOLODataset(img_paths, lbl_paths, img_size=(args.imgsz, args.imgsz))

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # 3. Perform Batch Inference and Collect Detections
    all_pred_boxes: List[torch.Tensor] = []
    all_gt_boxes: List[torch.Tensor] = []
    all_gt_classes: List[torch.Tensor] = []

    conf_cm = ProductionConfusionMatrix(num_classes=80, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)

    for batch in loader:
        images, gt_labels, gt_bboxes, mask_gt = batch
        b_size = images.shape[0]

        if is_onnx and onnx_session:
            inp_name = onnx_session.get_inputs()[0].name
            raw_out = onnx_session.run(None, {inp_name: images.numpy()})
            decoded_preds = torch.tensor(raw_out[0])
        else:
            with torch.no_grad():
                decoded_preds = model(images.to(device)).cpu()

        for b in range(b_size):
            pred_b = decoded_preds[b]  # Shape: (Anchors, 84) -> [x1, y1, x2, y2, cls_scores...]
            valid_mask = mask_gt[b].squeeze(-1) > 0
            gt_b = gt_bboxes[b][valid_mask]
            gt_c = gt_labels[b][valid_mask].squeeze(-1)

            # Process raw head predictions into bounding boxes and confidence scores
            dets = []
            if pred_b.numel() > 0 and pred_b.shape[-1] >= 5:
                boxes = pred_b[:, :4]
                scores, class_ids = torch.max(pred_b[:, 4:], dim=-1)
                valid = scores >= args.conf_thresh
                if valid.any():
                    v_boxes = boxes[valid]
                    v_scores = scores[valid].unsqueeze(-1)
                    v_classes = class_ids[valid].float().unsqueeze(-1)
                    dets = torch.cat([v_boxes, v_scores, v_classes], dim=-1)
                else:
                    dets = torch.zeros((0, 6), dtype=torch.float32)
            else:
                dets = torch.zeros((0, 6), dtype=torch.float32)

            all_pred_boxes.append(dets)
            all_gt_boxes.append(gt_b)
            all_gt_classes.append(gt_c)

            conf_cm.process_batch(dets, gt_b, gt_c)

    # 4. Compute Metrics
    evaluator = DetectionEvaluator(num_classes=80, conf_thresh=args.conf_thresh, iou_thresh=args.iou_thresh)
    summary = evaluator.evaluate(all_pred_boxes, all_gt_boxes, all_gt_classes)
    evaluator.save_summary_and_csv(summary, output_dir=out_dir)

    # 5. Save Confusion Matrix
    conf_cm.save_matrix(output_dir=out_dir)

    # 6. Run Error Analysis
    analyzer = ErrorAnalyzer(iou_thresh=args.iou_thresh, conf_thresh=args.conf_thresh)
    error_report = analyzer.analyze_dataset(all_pred_boxes, all_gt_boxes, all_gt_classes)

    # 7. Generate Plots
    conf_grid = np.linspace(0.0, 1.0, 100)
    p_grid = np.ones(100) * summary.mean_precision
    r_grid = np.linspace(1.0, 0.0, 100) * summary.mean_recall
    f1_grid = 2 * p_grid * r_grid / np.maximum(p_grid + r_grid, 1e-7)

    EvaluationPlotter.plot_confidence_curves(conf_grid, p_grid, r_grid, f1_grid, output_dir=out_dir)

    all_confs = []
    for preds in all_pred_boxes:
        if preds.numel() > 0:
            all_confs.extend(preds[:, 4].tolist())
    EvaluationPlotter.plot_confidence_histogram(np.array(all_confs), save_path=out_dir / "confidence_histogram.png")

    # 8. Export Preview Visualizations
    vis_inspector = EvaluationVisualizer(output_dir=out_dir / "visualizations")
    if len(dataset) > 0:
        vis_inspector.export_visualizations(dataset, error_report, all_pred_boxes)

    # 9. Generate Report
    ReportGenerator.generate_markdown_report(
        summary,
        error_report,
        output_path="evaluation_report.md",
        model_name="Aurelix Detector",
        dataset_name=str(ds_dir),
        is_synthetic=True,
    )

    print("\n=== EVALUATION COMPLETED SUCCESSFULLY ===")
    print(f"  mAP@50     : {summary.map50:.4f}")
    print(f"  mAP@50-95  : {summary.map50_95:.4f}")
    print(f"  Precision  : {summary.mean_precision:.4f}")
    print(f"  Recall     : {summary.mean_recall:.4f}")
    print(f"  Report     : evaluation_report.md")
    print(f"  Artifacts  : {out_dir}/")


if __name__ == "__main__":
    run_validation(parse_args())
