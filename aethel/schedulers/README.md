# Aurelix Schedulers Submodule (`aurelix.schedulers`)

## Purpose
The **Schedulers Submodule** manages learning rate decay curves, warmup schedules, and dynamic parameter adaptation over training epochs.

## Supported Schedulers

- **CosineAnnealingLR**: Cosine decay with minimum learning rate floor.
- **OneCycleLR**: Single cycle learning rate policy for fast convergence.
- **StepLR**: Stepwise multiplicative learning rate decay.
- **PolynomialLR**: Polynomial decay policy.

## Usage Example

```python
from aurelix.schedulers import SchedulerBuilder

# Build Cosine Annealing scheduler from config
builder = SchedulerBuilder(name="CosineAnnealingLR")
scheduler = builder.build(optimizer, epochs=100, min_lr=1e-5)
```
