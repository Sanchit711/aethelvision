# Aurelix Callbacks Submodule (`aurelix.callbacks`)

## Purpose
The **Callbacks Submodule** provides event hook abstractions for monitoring training events, logging metrics, saving checkpoints, and triggering early stopping.

## Event Hook Lifecycle

$$\text{on\_train\_start} \to \text{on\_epoch\_start} \to \text{on\_batch\_start} \to \text{on\_batch\_end} \to \text{on\_validation\_start} \to \text{on\_validation\_end} \to \text{on\_epoch\_end} \to \text{on\_train\_end}$$

## Available Callbacks

- **`ProgressLoggerCallback`**: Formats and logs epoch progress to stdout and log files.
- **`EarlyStoppingCallback`**: Stops training early when validation metrics stop improving.
- **`ModelCheckpointCallback`**: Triggers model weight saving on metric improvement.
- **`CallbackManager`**: Event dispatcher managing registered callback lists.

## Usage Example

```python
from aurelix.callbacks import CallbackManager, ProgressLoggerCallback, EarlyStoppingCallback

manager = CallbackManager([
    ProgressLoggerCallback(),
    EarlyStoppingCallback(monitor="val_loss", patience=10)
])
```
