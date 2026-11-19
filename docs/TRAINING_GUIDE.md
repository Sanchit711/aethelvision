# Mhyra Vision Framework — Training Guide

This guide details training model architectures using declarative YAML recipes and the generic `mhyra.trainer.Trainer`.

## Training Command Example

```python
from mhyra.configs.manager import ConfigManager
from mhyra.models import build_model
from mhyra.trainer import Trainer

# Load training recipe specification
config = ConfigManager.load_yaml("mhyra/configs/recipes/coco.yaml")

# Instantiate model
model = build_model(config.recipe.model, num_classes=config.recipe.num_classes)

# Instantiate generic trainer orchestrator
trainer = Trainer(config=config.recipe.__dict__, model=model)

# Run end-to-end model fitting
summary = trainer.fit()
```
