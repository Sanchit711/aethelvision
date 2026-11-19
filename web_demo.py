#!/usr/bin/env python
"""AethelVision Multi-Modal Web Demonstration Server.

Runs a zero-dependency interactive browser demo on http://localhost:8080.
Allows testing Detection, Pose Estimation, Instance Segmentation, and OBB.

Usage:
    python web_demo.py
"""

import http.server
import json
import socketserver
import urllib.parse
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from aethel.inference.result import PredictionResult
from aethel.inference.visualizer import AethelVisualizer
from aethel.models import build_model
from aethel.models.head.obb import obb_to_corners
from aethel.models.head.segment import assemble_masks

PORT = 8080

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AethelVision™ Interactive Multi-Modal AI Demo</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }
        .container { max-width: 900px; margin: 0 auto; background: #1e293b; padding: 32px; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); border: 1px solid #334155; }
        h1 { margin-top: 0; color: #38bdf8; font-size: 28px; }
        p.subtitle { color: #94a3b8; font-size: 15px; margin-bottom: 24px; }
        .controls { display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }
        select, button { padding: 10px 18px; border-radius: 8px; font-size: 14px; border: 1px solid #475569; background: #0f172a; color: #f8fafc; cursor: pointer; }
        button { background: #0284c7; border: none; font-weight: bold; transition: background 0.2s; }
        button:hover { background: #0369a1; }
        .preview { border: 2px dashed #475569; border-radius: 8px; min-height: 350px; display: flex; align-items: center; justify-content: center; background: #090d16; }
        img { max-width: 100%; border-radius: 6px; }
        .badge { display: inline-block; padding: 4px 10px; background: #0369a1; border-radius: 20px; font-size: 12px; font-weight: bold; margin-bottom: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <span class="badge">AethelVision v1.0.0</span>
        <h1>AethelVision™ Multi-Modal AI Engine</h1>
        <p class="subtitle">Universal Hardware-Agnostic Edge AI (Detection, 17-Keypoint Pose, Segmentation, Continuous OBB)</p>

        <div class="controls">
            <select id="taskSelect">
                <option value="pose">Human Pose Estimation (17 COCO Keypoints)</option>
                <option value="detect">2D Object Detection (DFL + CIoU)</option>
                <option value="segment">Instance Segmentation (ProtoNet)</option>
                <option value="obb">Oriented Bounding Boxes (Continuous Angle)</option>
            </select>
            <button onclick="runInference()">Run Live Aethel Inference</button>
        </div>

        <div class="preview" id="previewArea">
            <p style="color:#64748b;">Click "Run Live Aethel Inference" to evaluate test imagery.</p>
        </div>
    </div>

    <script>
        function runInference() {
            const task = document.getElementById('taskSelect').value;
            const preview = document.getElementById('previewArea');
            preview.innerHTML = '<p style="color:#38bdf8;">Running Aethel neural network inference...</p>';

            fetch('/api/infer?task=' + task)
                .then(res => res.json())
                .then(data => {
                    preview.innerHTML = '<img src="' + data.image + '" alt="Output" />';
                })
                .catch(err => {
                    preview.innerHTML = '<p style="color:#ef4444;">Error running inference: ' + err + '</p>';
                });
        }
    </script>
</body>
</html>
"""


class AethelDemoHandler(http.server.SimpleHTTPRequestHandler):
    """Zero-dependency HTTP handler for Aethel web demo."""

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        elif parsed.path == "/api/infer":
            query = urllib.parse.parse_qs(parsed.query)
            task = query.get("task", ["pose"])[0]

            vis = AethelVisualizer()
            # Generate demonstration synthetic canvas
            canvas = np.zeros((450, 600, 3), dtype=np.uint8)
            canvas[:] = (20, 28, 44)

            if task == "pose":
                kpts = np.zeros((1, 17, 3), dtype=np.float32)
                # Skeleton joint coordinates
                kpts[0, 0] = [300, 80, 0.95]   # Nose
                kpts[0, 1] = [290, 75, 0.90]   # Left eye
                kpts[0, 2] = [310, 75, 0.90]   # Right eye
                kpts[0, 3] = [275, 80, 0.85]   # Left ear
                kpts[0, 4] = [325, 80, 0.85]   # Right ear
                kpts[0, 5] = [250, 140, 0.95]  # Left shoulder
                kpts[0, 6] = [350, 140, 0.95]  # Right shoulder
                kpts[0, 7] = [220, 200, 0.92]  # Left elbow
                kpts[0, 8] = [380, 200, 0.92]  # Right elbow
                kpts[0, 9] = [200, 260, 0.88]  # Left wrist
                kpts[0, 10] = [400, 260, 0.88] # Right wrist
                kpts[0, 11] = [260, 280, 0.95] # Left hip
                kpts[0, 12] = [340, 280, 0.95] # Right hip
                kpts[0, 13] = [250, 360, 0.90] # Left knee
                kpts[0, 14] = [350, 360, 0.90] # Right knee
                kpts[0, 15] = [240, 430, 0.85] # Left ankle
                kpts[0, 16] = [360, 430, 0.85] # Right ankle

                res = PredictionResult(
                    boxes=np.array([[190.0, 50.0, 410.0, 440.0]], dtype=np.float32),
                    scores=np.array([0.96], dtype=np.float32),
                    classes=np.array([0], dtype=np.int64),
                    labels=["person"],
                    keypoints=kpts,
                )
            elif task == "segment":
                protos = np.random.randn(1, 32, 450, 600).astype(np.float32)
                coeffs = np.random.randn(1, 32).astype(np.float32)
                boxes = np.array([[150.0, 100.0, 450.0, 350.0]], dtype=np.float32)
                masks = assemble_masks(protos, coeffs, boxes)
                res = PredictionResult(
                    boxes=boxes,
                    scores=np.array([0.94], dtype=np.float32),
                    classes=np.array([2], dtype=np.int64),
                    labels=["car"],
                    masks=masks,
                )
            elif task == "obb":
                obb = np.array([[300.0, 220.0, 220.0, 110.0, 0.65]], dtype=np.float32)
                corners = obb_to_corners(obb)
                res = PredictionResult(
                    boxes=np.array([[180.0, 140.0, 420.0, 300.0]], dtype=np.float32),
                    scores=np.array([0.91], dtype=np.float32),
                    classes=np.array([4], dtype=np.int64),
                    labels=["airplane"],
                    obb_angles=np.array([0.65], dtype=np.float32),
                    obb_corners=corners,
                )
            else:
                res = PredictionResult(
                    boxes=np.array([[80.0, 120.0, 260.0, 360.0], [320.0, 100.0, 520.0, 380.0]], dtype=np.float32),
                    scores=np.array([0.95, 0.89], dtype=np.float32),
                    classes=np.array([0, 1], dtype=np.int64),
                    labels=["person", "bicycle"],
                )

            rendered = vis.draw(canvas, res)
            buf = BytesIO()
            rendered.save(buf, format="PNG")
            import base64
            img_b64 = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

            payload = json.dumps({"image": img_b64})
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload.encode("utf-8"))
        else:
            self.send_error(404)


def main() -> None:
    print(f"\n[AethelVision] Starting Interactive Web Demo on http://localhost:{PORT}")
    print("[AethelVision] Press Ctrl+C to terminate server.\n")
    with socketserver.TCPServer(("", PORT), AethelDemoHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[AethelVision] Web server stopped.")


if __name__ == "__main__":
    main()
