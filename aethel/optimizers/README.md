# Aurelix Optimizers Submodule (`aurelix.optimizers`)

## Purpose
The **Optimizers Submodule** defines builder factories for instantiating model parameter optimizers.

## Supported Optimizers

- **SGD**: Stochastic Gradient Descent with momentum.
- **Adam**: Adaptive Moment Estimation.
- **AdamW**: Adam with decoupled weight decay (default choice).
- **RMSprop**: Root Mean Square Propagation.

## Usage Example

```python
from aurelix.optimizers import OptimizerBuilder

# Build AdamW optimizer from config
builder = OptimizerBuilder(name="AdamW")
optimizer = builder.build(model.parameters(), lr=0.001, weight_decay=0.0005)
```
