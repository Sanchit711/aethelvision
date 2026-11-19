"""AETHEL Enterprise AI Vision Platform — Internal AI Assistant Subsystem.

Provides automated intelligent diagnostics, error explanation, fix recommendations,
dataset analysis, hyperparameter tuning recommendations, augmentation strategies,
and environment optimization advice.
"""

from typing import Any, Dict, List, Optional


class AethelAIAssistant:
    """Internal AI Vision Assistant & Diagnostic Advisor."""

    def __init__(self, root_dir: Optional[Any] = None) -> None:
        self.root_dir = root_dir

    def explain_error(self, error_message: str, traceback_str: str = "") -> Dict[str, Any]:
        """Analyze exception stack trace and generate root cause explanation + step-by-step fix."""
        err_lower = error_message.lower()
        
        if "cuda" in err_lower or "out of memory" in err_lower or "vram" in err_lower:
            return {
                "category": "GPU VRAM OOM Exception",
                "explanation": "The PyTorch CUDA runtime ran out of available VRAM allocated on the GPU.",
                "severity": "HIGH",
                "recommended_fixes": [
                    "Reduce training batch size from current value to batch_size=8 or batch_size=4.",
                    "Enable Automatic Mixed Precision (AMP) fp16 training mode.",
                    "Use gradient accumulation steps to maintain effective batch size.",
                    "Lower target image resolution from 640x640 to 512x512 or 416x416.",
                ],
            }
        elif "dataset" in err_lower or "yaml" in err_lower or "not found" in err_lower:
            return {
                "category": "Dataset Configuration / File I/O Error",
                "explanation": "The specified dataset path or data.yaml structure could not be parsed.",
                "severity": "MEDIUM",
                "recommended_fixes": [
                    "Verify that data.yaml exists inside the dataset directory.",
                    "Ensure train/val image paths are relative to dataset root or absolute.",
                    "Run Dataset Center -> [6] Dataset Validation to verify directory layout.",
                ],
            }
        else:
            return {
                "category": "Runtime System Exception",
                "explanation": f"An unhandled system exception was captured: {error_message}",
                "severity": "MEDIUM",
                "recommended_fixes": [
                    "Inspect full error log stored at logs/launcher_errors.log.",
                    "Check system memory and disk space availability.",
                    "Run System Center -> [14] Diagnostics to run integrity checks.",
                ],
            }

    def analyze_dataset_health(self, dataset_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dataset distribution and recommend data engineering actions."""
        train_cnt = dataset_stats.get("train_imgs", 0)
        val_cnt = dataset_stats.get("val_imgs", 0)
        num_classes = dataset_stats.get("num_classes", 1)

        recommendations = []
        health_score = 95.0

        if val_cnt == 0 or (train_cnt > 0 and (val_cnt / train_cnt) < 0.1):
            recommendations.append("Validation split is under 10% of training dataset. Recommended split is 80/20.")
            health_score -= 15.0

        if train_cnt < 100 * max(1, num_classes):
            recommendations.append(f"Image volume ({train_cnt} imgs) is low for {num_classes} classes. Consider adding data augmentation or fine-tuning from pre-trained weights.")
            health_score -= 10.0

        if not recommendations:
            recommendations.append("Dataset balance and split proportions are HEALTHY and optimal for training.")

        return {
            "health_score": max(50.0, health_score),
            "status": "HEALTHY" if health_score >= 80.0 else "WARNING",
            "recommendations": recommendations,
        }

    def recommend_hyperparameters(self, dataset_info: Dict[str, Any], hw_info: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal hyperparameter set based on GPU specs and dataset size."""
        vram_mb = hw_info.get("vram_free_mb", 2048)
        cuda_ok = hw_info.get("cuda_available", False)

        if cuda_ok and vram_mb > 6000:
            batch_size = 32
            img_size = 640
            amp = True
        elif cuda_ok and vram_mb > 3000:
            batch_size = 16
            img_size = 640
            amp = True
        else:
            batch_size = 8
            img_size = 512
            amp = False

        return {
            "recommended_batch_size": batch_size,
            "recommended_img_size": img_size,
            "recommended_lr": 0.001,
            "recommended_optimizer": "AdamW",
            "recommended_scheduler": "CosineAnnealing",
            "enable_amp": amp,
            "recommended_augmentations": ["RandomFlip", "ColorJitter", "AffineScaling", "Mosaic"],
        }


__all__ = ["AethelAIAssistant"]
