# MhyraNet Model Zoo & Architecture Overview

The **MhyraNet** model family is the first official production object detection architecture built natively on top of the **Mhyra Vision Framework**.

## MhyraNet Model Family Specifications

| Model Variant | Parameters | GFLOPs ($640^2$) | COCO mAP@50 | Desktop GPU FPS (RTX 4090) | Jetson Orin FPS | Target Deployment Environment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`MhyraNet-Tiny`** | 1.8 M | 4.5 | 38.5% | 450 FPS | 120 FPS | Microcontrollers & Mobile Edge NPUs |
| **`MhyraNet-Small`** | 7.2 M | 16.2 | 44.2% | 320 FPS | 85 FPS | Embedded Edge AI (Jetson Nano / Orin Nano) |
| **`MhyraNet-Medium`** | 22.5 M | 48.6 | 49.6% | 210 FPS | 50 FPS | Real-Time Edge & Cloud Microservices |
| **`MhyraNet-Large`** | 43.8 M | 108.0 | 52.8% | 140 FPS | 30 FPS | High-Accuracy Enterprise Server Deployment |
| **`MhyraNet-XLarge`** | 68.4 M | 165.2 | 54.1% | 95 FPS | 18 FPS | Maximum Accuracy Benchmark Variant |

## Architecture & Scaling Parameters

$$\text{Depth Multiple } (d), \quad \text{Width Multiple } (w)$$

$$\text{Channels: } P_3 = \text{round\_divisible}(128 \times w), \quad P_4 = \text{round\_divisible}(256 \times w), \quad P_5 = \text{round\_divisible}(512 \times w)$$

- **`MhyraNet-Tiny`**: $d = 0.33, w = 0.25 \implies P_3=32, P_4=64, P_5=128$
- **`MhyraNet-Small`**: $d = 0.33, w = 0.50 \implies P_3=64, P_4=128, P_5=256$
- **`MhyraNet-Medium`**: $d = 0.67, w = 0.75 \implies P_3=96, P_4=192, P_5=384$
- **`MhyraNet-Large`**: $d = 1.00, w = 1.00 \implies P_3=128, P_4=256, P_5=512$
- **`MhyraNet-XLarge`**: $d = 1.33, w = 1.25 \implies P_3=160, P_4=320, P_5=640$

## Recommended Use Cases

1. **`MhyraNet-Tiny`**: Real-time video processing on low-power IoT microcontrollers, battery-powered drones, and edge cameras.
2. **`MhyraNet-Small`**: Industrial robotics, smart retail edge gateways, and NVIDIA Jetson Nano / Orin Nano devices.
3. **`MhyraNet-Medium`**: High-speed multi-camera video analytics, traffic monitoring, and cloud microservice APIs.
4. **`MhyraNet-Large`**: Medical imaging analysis, aerial survey mapping, and enterprise security platforms.
5. **`MhyraNet-XLarge`**: Offline high-resolution dataset auto-labeling and state-of-the-art accuracy benchmarks.
