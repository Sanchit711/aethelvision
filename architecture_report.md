# Architecture Audit Report

## MhyraDetector Structural Summary
- **Backbone**: `DetectorStem` + 3 `CSPStage` blocks with `CBAM` spatial/channel attention
- **Neck**: `PAFPN` (Path Aggregation Feature Pyramid Network) across scales P3, P4, P5
- **Head**: `DecoupledDetectHead` anchor-free classification & bounding box regression branches
- **Decoder**: `HeadDecoder` with Distribution Focal Loss (`reg_max=16`)
- **Loss Subsystem**: `CompositeDetectionLoss` (CIoU + DFL + Focal Classification Loss)

## Parameter Count Metrics
- **Total Parameters**: `0`
- **Trainable Parameters**: `0`
- **Frozen Parameters**: `0`
