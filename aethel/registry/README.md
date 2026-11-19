# Aurelix Registry Subsystem (`aurelix.registry`)

## Purpose
The **Registry Subsystem** provides a framework-independent, decoupled mechanism for mapping string names to classes, factories, or functions. It enables dynamic component instantiation from configuration files without tight code coupling.

## Responsibilities
- Maintain pre-instantiated registries for datasets, models, backbones, necks, heads, losses, metrics, trainers, exporters, predictors, optimizers, schedulers, callbacks, and visualizers.
- Provide a `@registry.register()` decorator pattern for zero-boilerplate module discovery.
- Support key validation, duplicate registration guards, and isolated test clearing capabilities.

## Public API
- `Registry`: Generic registry container class.
- `DATASET_REGISTRY`, `MODEL_REGISTRY`, `LOSS_REGISTRY`, `TRAINER_REGISTRY`, `METRIC_REGISTRY`, `EXPORTER_REGISTRY`, `PREDICTOR_REGISTRY`, `OPTIMIZER_REGISTRY`, `SCHEDULER_REGISTRY`, `CALLBACK_REGISTRY`, `VISUALIZER_REGISTRY`.

## Dependencies
- Standard Python libraries (`typing`, `Callable`, `Dict`).
- `aurelix.utils.exceptions.RegistryException`
- `aurelix.utils.logger`

## Future Work
- Support dynamic package auto-discovery and plugin entry-point scanning.
- Add strict type checking over registered interface contracts.
