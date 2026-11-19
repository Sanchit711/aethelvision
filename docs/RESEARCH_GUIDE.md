# Mhyra Vision Framework — AI Research Guide

Welcome to the **Mhyra Vision Framework AI Research Platform (v1.1)**. This guide outlines procedures for extending backbone architectures, custom loss functions, target assigners, and evaluation metrics.

## Key Research Subsystems

1. **Model Architecture Extensibility**: Inherit `mhyra.models.BaseModel` and decorate with `@MODEL_REGISTRY.register("YourModel")`.
2. **Custom Loss Formulations**: Inherit `mhyra.losses.BaseLoss` and register via `@LOSS_REGISTRY.register("YourLoss")`.
3. **Experiment Tracking**: Use `mhyra.trainer.ExperimentManager` to record CSV metrics and JSON training summaries automatically.
