# Aurelix Layers Submodule (`aurelix.layers`)

## Purpose
The **Layers Submodule** maintains reusable, atomic neural network building blocks (convolutions, spatial attention modules, bottleneck stages, feature fusion blocks).

## Responsibilities
- Implement standardized convolutional blocks (`Conv`, `DWConv`).
- Implement spatial and channel attention primitives (`CBAM`, `SEBlock`).
- Implement residual stage blocks (`CSPStage`, `RepVGGBlock`, `SPPF`).

## Public API
- Custom `torch.nn.Module` layer primitives registered in `LAYER_REGISTRY`.

## Dependencies
- `torch`, `torch.nn`
- `aurelix.registry.LAYER_REGISTRY`

## Future Work
- Add custom C++/CUDA kernel bindings for low-latency layer operations.
