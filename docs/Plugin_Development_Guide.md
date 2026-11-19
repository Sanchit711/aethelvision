# AURELIX Enterprise AI Vision Platform — Plugin Development Guide

**Version**: 1.0.0 Commercial Release  

---

## 1. Plugin Architecture
AURELIX plugins extend platform capabilities across 7 domain categories:
1. **Detection**: Custom detection heads and NMS algorithms.
2. **Tracking**: Multi-object tracking (ByteTrack, DeepSORT).
3. **OCR**: Optical character recognition & text line orientation.
4. **Segmentation**: Mask & polygon instance segmentation.
5. **Pose**: Skeletal keypoint estimation.
6. **Classification**: Deep image classifiers.
7. **Custom**: Analytics, geofencing, and pipeline sinks.

---

## 2. Plugin Structure
Create a directory inside `plugins/<plugin_name>/` containing `plugin.json`:

```json
{
  "id": "com.custom.detector",
  "name": "Custom Industrial Detector",
  "type": "Detection",
  "version": "1.0.0",
  "status": "ACTIVE",
  "description": "Industrial defect object detection plugin.",
  "author": "Enterprise Systems Corp"
}
```

AURELIX automatically discovers all compliant plugin manifests during startup.

---
*(c) 2026 AURELIX Technologies Inc. All Rights Reserved.*
