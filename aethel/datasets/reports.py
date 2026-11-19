"""AETHEL Dataset Reports Generator.

Generates:
- runs/datasets/index.json
- runs/datasets/history.json
- runs/datasets/report.html (Interactive HTML report with SVG charts and health metrics)
- runs/datasets/report.csv
"""

import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from aethel.datasets.discovery import DynamicDatasetScanner, format_size_bytes


class DatasetReportGenerator:
    """Generates canonical dataset indexes and HTML/CSV reports in runs/datasets/."""

    def __init__(self, root_dir: Path = Path(".")) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.output_dir = self.root_dir / "runs" / "datasets"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scanner = DynamicDatasetScanner(root_dir=self.root_dir)

    def generate_all_reports(self) -> Dict[str, str]:
        """Discover live datasets and generate index.json, history.json, report.html, report.csv."""
        datasets = self.scanner.discover_datasets(force_rescan=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # 1. runs/datasets/index.json
        index_path = self.output_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(datasets, f, indent=2)

        # 2. runs/datasets/history.json
        history_path = self.output_dir / "history.json"
        history_records = []
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history_records = json.load(f)
            except Exception:
                pass

        history_records.append({
            "timestamp": timestamp,
            "total_datasets": len(datasets),
            "datasets": [
                {
                    "name": d["name"],
                    "format": d["format"],
                    "health_score": d["health_score"],
                    "status": d["status"],
                    "total_imgs": d["total_imgs"],
                }
                for d in datasets
            ],
        })

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_records, f, indent=2)

        # 3. runs/datasets/report.csv
        csv_path = self.output_dir / "report.csv"
        fieldnames = [
            "name", "format", "num_classes", "train_imgs", "val_imgs", "test_imgs",
            "total_imgs", "size_str", "health_score", "status", "display_status",
            "corrupted_images", "duplicate_images", "missing_labels", "last_modified"
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in datasets:
                row = {k: d.get(k, "") for k in fieldnames}
                writer.writerow(row)

        # 4. runs/datasets/report.html
        html_path = self.output_dir / "report.html"
        self._generate_html_report(html_path, datasets, timestamp)

        return {
            "index.json": str(index_path),
            "history.json": str(history_path),
            "report.csv": str(csv_path),
            "report.html": str(html_path),
        }

    def _generate_html_report(self, html_path: Path, datasets: List[Dict[str, Any]], timestamp: str) -> None:
        """Generate standalone HTML dataset report with SVG charts and health dashboard."""
        total_ds = len(datasets)
        total_imgs = sum(d.get("total_imgs", 0) for d in datasets)
        total_size = sum(d.get("size_bytes", 0) for d in datasets)
        total_size_str = format_size_bytes(total_size)
        avg_health = round(sum(d.get("health_score", 100) for d in datasets) / max(1, total_ds), 1)

        # Build table rows HTML
        rows_html = ""
        for i, d in enumerate(datasets):
            h_color = "#10b981" if d["health_score"] >= 90 else ("#f59e0b" if d["health_score"] >= 70 else "#ef4444")
            s_color = "#10b981" if d["status"] == "READY" else ("#f59e0b" if d["status"] == "WARNING" else "#ef4444")
            rows_html += f"""
            <tr>
                <td><strong>{i + 1}</strong></td>
                <td><strong>{d['name']}</strong><br><small style="color: #6b7280;">{d['rel_path']}</small></td>
                <td><span class="badge" style="background: #1e293b; color: #38bdf8;">{d['format']}</span></td>
                <td>{d['num_classes']}</td>
                <td>{d['train_imgs']}</td>
                <td>{d['val_imgs']}</td>
                <td>{d['test_imgs']}</td>
                <td>{d['size_str']}</td>
                <td><strong style="color: {h_color};">{d['health_score']}%</strong></td>
                <td><span class="badge" style="background: {s_color}22; color: {s_color}; border: 1px solid {s_color};">{d['display_status']}</span></td>
            </tr>
            """

        # SVG Bar chart for dataset sizes
        chart_bars = ""
        max_size = max([d.get("size_bytes", 1) for d in datasets] or [1])
        for i, d in enumerate(datasets[:8]):
            h_pct = max(10, int((d.get("size_bytes", 0) / max_size) * 180))
            x_pos = 50 + i * 80
            y_pos = 220 - h_pct
            chart_bars += f"""
            <rect x="{x_pos}" y="{y_pos}" width="40" height="{h_pct}" rx="4" fill="#38bdf8"/>
            <text x="{x_pos + 20}" y="240" font-size="11" fill="#94a3b8" text-anchor="middle">{d['name'][:8]}</text>
            """

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AETHEL Enterprise Dataset Health Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 24px; }}
        .header {{ display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #334155; padding-bottom: 16px; margin-bottom: 24px; }}
        .title {{ font-size: 24px; font-weight: bold; color: #38bdf8; }}
        .meta {{ font-size: 14px; color: #94a3b8; }}
        .cards {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
        .card {{ background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; }}
        .card-val {{ font-size: 28px; font-weight: bold; color: #f8fafc; margin-top: 8px; }}
        .card-lbl {{ font-size: 13px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; }}
        .section {{ background: #1e293b; padding: 20px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 24px; }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ padding: 12px; background: #0f172a; color: #94a3b8; font-size: 13px; border-bottom: 2px solid #334155; }}
        td {{ padding: 12px; border-bottom: 1px solid #334155; font-size: 14px; }}
        .badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; display: inline-block; }}
    </style>
</head>
<body>
    <div class="header">
        <div>
            <div class="title">AETHEL Enterprise Dataset Health Report</div>
            <div class="meta">Live Filesystem Audit & Health Diagnostics</div>
        </div>
        <div class="meta">Generated: {timestamp}</div>
    </div>

    <div class="cards">
        <div class="card">
            <div class="card-lbl">Total Datasets</div>
            <div class="card-val">{total_ds}</div>
        </div>
        <div class="card">
            <div class="card-lbl">Total Images</div>
            <div class="card-val">{total_imgs:,}</div>
        </div>
        <div class="card">
            <div class="card-lbl">Total Disk Storage</div>
            <div class="card-val">{total_size_str}</div>
        </div>
        <div class="card">
            <div class="card-lbl">Average Health Score</div>
            <div class="card-val" style="color: #10b981;">{avg_health}%</div>
        </div>
    </div>

    <div class="section">
        <h3>Dataset Size Distribution</h3>
        <svg width="700" height="260">
            <line x1="40" y1="220" x2="680" y2="220" stroke="#334155" stroke-width="2"/>
            {chart_bars}
        </svg>
    </div>

    <div class="section">
        <h3>Dynamic Dataset Inventory</h3>
        <table>
            <thead>
                <tr>
                    <th>#</th>
                    <th>Dataset Name</th>
                    <th>Format</th>
                    <th>Classes</th>
                    <th>Train</th>
                    <th>Val</th>
                    <th>Test</th>
                    <th>Size</th>
                    <th>Health</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</body>
</html>
"""
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
