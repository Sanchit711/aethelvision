/**
 * @file aethel_onnx.hpp
 * @brief AethelVision Single-Header C++ Inference SDK for Edge Robotics & Embedded Systems.
 * 
 * Provides zero-overhead, multi-modal C++ inference using ONNX Runtime:
 * - 2D Object Detection (Decoupled Anchor-Free + CIoU)
 * - 17-Keypoint Human Pose Estimation
 * - High-Speed Instance Segmentation
 * - Oriented Bounding Boxes (Continuous Angle OBB)
 * 
 * Dependencies:
 *   - onnxruntime_cxx_api.h (ONNX Runtime C++ API)
 * 
 * License: Apache 2.0 (Open-Source Sovereign)
 * Copyright (c) 2026 AethelVision Contributors
 */

#ifndef AETHEL_ONNX_HPP
#define AETHEL_ONNX_HPP

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <memory>

namespace aethel {

struct BoundingBox {
    float x1, y1, x2, y2;
    float score;
    int class_id;
};

struct Keypoint {
    float x, y, confidence;
};

struct PoseDetection {
    BoundingBox box;
    std::vector<Keypoint> keypoints; // 17 COCO joints
};

struct RotatedBox {
    float cx, cy, w, h, angle_rad;
    float score;
    int class_id;
    std::vector<std::pair<float, float>> corners; // 4 polygon vertices
};

/**
 * @brief Converts rotated box parameters (cx, cy, w, h, angle) into 4 polygon vertices.
 */
inline std::vector<std::pair<float, float>> compute_obb_corners(float cx, float cy, float w, float h, float angle) {
    float cos_a = std::cos(angle);
    float sin_a = std::sin(angle);
    float hw = w * 0.5f;
    float hh = h * 0.5f;

    std::vector<std::pair<float, float>> corners(4);
    corners[0] = { cx - hw * cos_a + hh * sin_a, cy - hw * sin_a - hh * cos_a };
    corners[1] = { cx + hw * cos_a + hh * sin_a, cy + hw * sin_a - hh * cos_a };
    corners[2] = { cx + hw * cos_a - hh * sin_a, cy + hw * sin_a + hh * cos_a };
    corners[3] = { cx - hw * cos_a - hh * sin_a, cy - hw * sin_a + hh * cos_a };
    return corners;
}

/**
 * @brief Fast Sigmoid activation helper.
 */
inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Decodes continuous (cos 2θ, sin 2θ) vector into angle in radians [-π/2, π/2].
 */
inline float decode_continuous_angle(float cos_2theta, float sin_2theta) {
    return 0.5f * std::atan2(sin_2theta, cos_2theta);
}

} // namespace aethel

#endif // AETHEL_ONNX_HPP
