"""Unit Tests for Restructured Aurelix Architecture Components.

Verifies registry functionality, configuration parsing, centralized logging,
exception hierarchy, and base interface contracts.
"""

import pytest
import logging
from pathlib import Path

from aurelix.utils.exceptions import (
    AurelixException,
    DatasetException,
    ModelException,
    TrainingException,
    ExportException,
    InferenceException,
    ConfigurationException,
    RegistryException,
)
from aurelix.utils.logger import get_logger, setup_logger
from aurelix.registry.registry import (
    Registry,
    MODEL_REGISTRY,
    DATASET_REGISTRY,
    LOSS_REGISTRY,
)
from aurelix.configs.manager import ConfigDict, ConfigManager
from aurelix.datasets.base import BaseDataset
from aurelix.models.base import BaseModel
from aurelix.losses.base import BaseLoss
from aurelix.trainer.base import BaseTrainer
from aurelix.metrics.base import BaseMetric
from aurelix.export.base import BaseExporter
from aurelix.inference.base import BasePredictor
from aurelix.optimizers.base import BaseOptimizer
from aurelix.schedulers.base import BaseScheduler
from aurelix.callbacks.base import BaseCallback
from aurelix.visualization.base import BaseVisualizer
from aurelix.engine.base import BaseEngine


def test_exception_hierarchy():
    """Verify exception inheritance and string formatting."""
    exc = DatasetException("Dataset loading failed", details="File not found")
    assert isinstance(exc, AurelixException)
    assert "Dataset loading failed" in str(exc)
    assert "File not found" in str(exc)


def test_logger_setup(tmp_path):
    """Verify structured logger creation and file output."""
    log_file = tmp_path / "test.log"
    logger = setup_logger(name="test_aurelix", log_file=log_file, level=logging.DEBUG)
    logger.info("Test log message")
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Test log message" in content


def test_registry_system():
    """Verify generic registry decorator, lookup, and duplicate guards."""
    test_reg = Registry("TEST_REGISTRY")

    @test_reg.register("custom_dummy")
    class DummyClass:
        pass

    assert test_reg.contains("custom_dummy")
    assert test_reg.get("custom_dummy") == DummyClass

    # Verify duplicate guard raises RegistryException
    with pytest.raises(RegistryException):
        @test_reg.register("custom_dummy")
        class DuplicateClass:
            pass


def test_config_manager(tmp_path):
    """Verify YAML configuration parsing and dot-notation access."""
    yaml_content = """
    model:
      name: "aurelix_detector"
      num_classes: 80
    """
    cfg_file = tmp_path / "model.yaml"
    cfg_file.write_text(yaml_content, encoding="utf-8")

    cfg = ConfigManager.load_yaml(cfg_file)
    assert isinstance(cfg, ConfigDict)
    assert cfg.model.name == "aurelix_detector"
    assert cfg.model.num_classes == 80


def test_base_interfaces():
    """Verify base interfaces cannot be instantiated directly without abstract implementations."""
    with pytest.raises(TypeError):
        BaseDataset(root="dummy")

    with pytest.raises(TypeError):
        BaseModel()

    with pytest.raises(TypeError):
        BaseLoss()

    with pytest.raises(TypeError):
        BaseTrainer(config={})

    with pytest.raises(TypeError):
        BaseMetric()

    with pytest.raises(TypeError):
        BaseExporter(model=None)

    with pytest.raises(TypeError):
        BasePredictor(model_path="dummy.pt")

    with pytest.raises(TypeError):
        BaseOptimizer()

    with pytest.raises(TypeError):
        BaseScheduler()

    with pytest.raises(TypeError):
        BaseVisualizer()

    with pytest.raises(TypeError):
        BaseEngine()
