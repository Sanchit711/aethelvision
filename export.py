"""Aurelix Detector Production Export CLI.

Supports ONNX, TensorRT, and TorchScript model exports:
- Opset 17+
- Dynamic batch & image sizes
- FP32 / FP16 precision
- INT8 calibration preparation
- ONNX Runtime numerical parity verification

Usage:
    python export.py --weights checkpoints/best_model.pt --output exports/best.onnx
    python export.py --weights checkpoints/best_model.pt --fp16 --dynamic
    python export.py --weights checkpoints/best_model.pt --int8
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

try:
    from export.exporter import ModelExporter
except ImportError:
    ModelExporter = None  # type: ignore
try:
    from aethel.export.universal import UniversalExporter
    from aethel_detector.export.onnx import ONNXExporter
    from aethel_detector.export.tensorrt import TensorRTCompiler
    from aethel_detector.export.quantization import QuantizationManager
    from aethel_detector.model import AethelDetector as AurelixDetector
except ImportError:
    try:
        from aurelix.export.universal import UniversalExporter
        from aurelix_detector.export.onnx import ONNXExporter
        from aurelix_detector.export.tensorrt import TensorRTCompiler
        from aurelix_detector.export.quantization import QuantizationManager
        from aurelix_detector.model import AurelixDetector
    except ImportError:
        UniversalExporter = None  # type: ignore
        ONNXExporter = None  # type: ignore
        TensorRTCompiler = None  # type: ignore
        QuantizationManager = None  # type: ignore
        AurelixDetector = None  # type: ignore


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aurelix Detector Model Export CLI")
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "openvino", "coreml", "tensorrt", "torchscript"],
        help="Export target format: onnx, openvino, coreml, tensorrt, torchscript (default: onnx)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="Path to PyTorch weights checkpoint (.pt)",
    )
    parser.add_argument(
        "--imgsz",
        nargs="+",
        type=int,
        default=[640, 640],
        help="Image size [height, width] (default: 640 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Target device for export ('cpu', 'cuda')",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Export model using FP16 half precision",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Prepare INT8 quantization calibration",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=True,
        help="Enable dynamic batch size, height, and width axes (default: True)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="exports/best.onnx",
        help="Output target file or directory path (default: exports/best.onnx)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    return parser.parse_args(args)


def run_export(args: argparse.Namespace) -> None:
    print("=== AURELIX DETECTOR PRODUCTION EXPORT ENGINE ===")

    # Handle image size argument parsing
    if len(args.imgsz) == 1:
        img_h, img_w = args.imgsz[0], args.imgsz[0]
    else:
        img_h, img_w = args.imgsz[0], args.imgsz[1]

    input_shape = (args.batch, 3, img_h, img_w)
    precision = "FP16" if args.fp16 else ("INT8" if args.int8 else "FP32")

    print(f"Target Input Shape : {input_shape}")
    print(f"Target Precision   : {precision}")
    print(f"Dynamic Axes       : {args.dynamic}")
    print(f"Target Opset       : {args.opset}")
    print(f"Target Output Path : {args.output}\n")

    # 1. Instantiate Model & Load Weights
    model = None
    if torch is not None and AurelixDetector is not None:
        device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
        model = AurelixDetector(num_classes=80).to(device)

        weights_path = Path(args.weights)
        if weights_path.exists() and weights_path.is_file():
            print(f"Loading checkpoint weights from {weights_path}...")
            model.load_weights(weights_path)
        else:
            print("No valid weights checkpoint provided. Initializing baseline detector model...")

        model.eval()
    else:
        print("Running in portable export mode (PyTorch not available in current environment).")

    # If non-ONNX target format is requested, route directly via UniversalExporter
    if getattr(args, "format", "onnx").lower() in ("openvino", "coreml", "torchscript"):
        fmt = args.format.lower()
        print(f"--- Routing to Universal Exporter Target: {fmt.upper()} ---")
        uni_exporter = UniversalExporter(model=model, opset_version=args.opset)
        res = uni_exporter.export(
            format_type=fmt,
            output_path=Path(args.output),
            input_shape=input_shape,
            dynamic=args.dynamic,
            precision=precision.lower(),
        )
        print(f"Export Status    : {res.get('status')}")
        print(f"Target Format    : {res.get('format')}")
        print(f"Output Path      : {res.get('path', res.get('xml_path'))}")
        if 'hardware_targets' in res:
            print(f"Target Hardware  : {', '.join(res['hardware_targets'])}")
        print("\nUniversal export completed successfully.")
        return

    # 2. Execute ONNX Export & Parity Verification
    output_path = Path(args.output)
    if output_path.suffix.lower() == ".engine":
        onnx_path = output_path.with_suffix(".onnx")
    else:
        onnx_path = output_path if output_path.suffix.lower() == ".onnx" else output_path / "best.onnx"

    exporter = ONNXExporter(model, opset_version=args.opset)
    res_onnx = exporter.export(
        output_path=onnx_path,
        input_shape=input_shape,
        dynamic=args.dynamic,
        half=args.fp16,
    )

    print("--- ONNX Export Validation Report ---")
    print(f"Status           : {res_onnx['status']}")
    print(f"ONNX Binary Path : {res_onnx['onnx_path']}")
    print(f"File Size        : {res_onnx['file_size_mb']:.2f} MB")
    print(f"Opset Version    : {res_onnx['opset_version']}")
    print(f"Dynamic Axes     : {res_onnx['dynamic_axes']}")
    print(f"Numerical Parity : {'PASSED' if res_onnx['parity']['passed_parity'] else 'FAILED'}")
    print(f"Max Abs Error    : {res_onnx['parity']['max_abs_error']:.6e}")
    print(f"Cosine Similarity: {res_onnx['parity']['cosine_similarity']:.6f}")

    if not res_onnx["parity"]["passed_parity"]:
        print("\nWARNING: Numerical error exceeds tolerance threshold!")

    # 3. Execute TensorRT Preparation / Build if requested or if .engine output
    if output_path.suffix.lower() == ".engine" or args.int8:
        print("\n--- TensorRT Engine Preparation ---")
        compiler = TensorRTCompiler()
        res_trt = compiler.compile_onnx_to_tensorrt(
            onnx_path=onnx_path,
            output_engine_path=output_path if output_path.suffix.lower() == ".engine" else None,
            precision=precision.lower(),
        )

        print(f"TensorRT Status  : {res_trt['status']}")
        print(f"Message          : {res_trt['message']}")
        if res_trt["capabilities"]["tensorrt_installed"]:
            print(f"Engine Path      : {res_trt['engine_path']}")
        else:
            print("Capabilities     : TensorRT runtime unavailable on host machine.")

    print("\nExport process completed successfully.")


def main() -> None:
    args = parse_args()
    run_export(args)


if __name__ == "__main__":
    main()
