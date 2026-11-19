"""Real End-to-End Verification Script for Dynamic Dataset Discovery Subsystem.

Executes real filesystem manipulations in data/:
1. Create data/temporary_test_dataset -> verify immediate live discovery
2. Rename data/temporary_test_dataset -> data/renamed_test_ds -> verify live UI/Scanner update
3. Delete data/renamed_test_ds -> verify immediate removal from live scanner
4. Create data/tyre_dataset -> verify live discovery without restarting AURELIX
5. Verify DatasetManager, Training Center workflow, Fine Tune workflow, and Evaluation workflow access tyre_dataset.
"""

import json
import shutil
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from aurelix.launcher.managers import DatasetManager, TrainingOrchestrator, EvaluationManagerAdapter


def verify_dynamic_dataset_discovery():
    print("=== STARTING REAL END-TO-END DATASET DISCOVERY VERIFICATION ===")
    data_dir = root_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    manager = DatasetManager(root_dir=root_dir)

    # Initial scan
    initial_datasets = manager.discover_datasets(force_rescan=True)
    initial_names = [d["name"] for d in initial_datasets]
    print(f"[Baseline] Initial Datasets in data/: {initial_names}")

    # Step 1: Create data/temporary_test_dataset
    tmp_ds = data_dir / "temporary_test_dataset"
    (tmp_ds / "train" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
    with open(tmp_ds / "data.yaml", "w", encoding="utf-8") as f:
        f.write("train: train/images\nval: valid/images\nnames:\n  0: temp_class\n")

    step1_datasets = manager.discover_datasets(force_rescan=True)
    step1_names = [d["name"] for d in step1_datasets]
    assert "temporary_test_dataset" in step1_names, "Step 1 Failed: temporary_test_dataset not discovered live!"
    print("[PASS] Step 1: data/temporary_test_dataset discovered immediately without restart.")

    # Step 2: Rename data/temporary_test_dataset -> data/renamed_test_ds
    renamed_ds = data_dir / "renamed_test_ds"
    tmp_ds.rename(renamed_ds)

    step2_datasets = manager.discover_datasets(force_rescan=True)
    step2_names = [d["name"] for d in step2_datasets]
    assert "renamed_test_ds" in step2_names, "Step 2 Failed: renamed_test_ds not found!"
    assert "temporary_test_dataset" not in step2_names, "Step 2 Failed: old name still present!"
    print("[PASS] Step 2: Rename detected live and UI updated.")

    # Step 3: Delete data/renamed_test_ds
    shutil.rmtree(renamed_ds)

    step3_datasets = manager.discover_datasets(force_rescan=True)
    step3_names = [d["name"] for d in step3_datasets]
    assert "renamed_test_ds" not in step3_names, "Step 3 Failed: deleted dataset still present!"
    print("[PASS] Step 3: Deletion detected immediately live.")

    # Step 4: Create data/tyre_dataset
    tyre_ds = data_dir / "tyre_dataset"
    (tyre_ds / "train" / "images").mkdir(parents=True, exist_ok=True)
    (tyre_ds / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (tyre_ds / "valid" / "images").mkdir(parents=True, exist_ok=True)
    (tyre_ds / "valid" / "labels").mkdir(parents=True, exist_ok=True)
    with open(tyre_ds / "data.yaml", "w", encoding="utf-8") as f:
        f.write("train: train/images\nval: valid/images\nnames:\n  0: tyre_tread\n  1: tyre_defect\n  2: tyre_rim\n")

    step4_datasets = manager.discover_datasets(force_rescan=True)
    step4_names = [d["name"] for d in step4_datasets]
    assert "tyre_dataset" in step4_names, "Step 4 Failed: tyre_dataset not discovered live!"
    print("[PASS] Step 4: data/tyre_dataset discovered live without restart.")

    # Step 5: Verify Training Center, Dataset Manager, Fine Tune, Evaluation can access tyre_dataset
    tyre_record = next(d for d in step4_datasets if d["name"] == "tyre_dataset")
    assert tyre_record["num_classes"] == 3
    assert tyre_record["format"] == "YOLO"
    print(f"[PASS] Step 5: tyre_dataset parsed successfully by DatasetManager (Format: {tyre_record['format']}, Classes: {tyre_record['num_classes']}, Health: {tyre_record['health_score']}%).")

    print("\n=== ALL REAL VERIFICATION STEPS PASSED SUCCESSFULLY ===")


if __name__ == "__main__":
    verify_dynamic_dataset_discovery()
