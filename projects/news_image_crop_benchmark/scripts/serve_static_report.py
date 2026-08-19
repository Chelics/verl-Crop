#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


REPORT_CSS = """
.gradio-container { max-width: none !important; padding: 0 !important; }
footer { display: none !important; }
.report-shell { height: calc(100vh - 74px); min-height: 720px; }
.report-shell iframe { width: 100%; height: 100%; border: 0; display: block; background: #f7f7f3; }
.report-toolbar { padding: 8px 14px; border-bottom: 1px solid #d7d9d3; }
"""


def validate_report_dir(report_dir: Path) -> Path:
    report_dir = report_dir.expanduser().resolve()
    required = ("report.html", "summary.json", "details.jsonl", "details.csv")
    missing = [name for name in required if not (report_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"report directory is missing required files: {missing}")
    return report_dir


def build_report_blocks(report_dir: Path) -> gr.Blocks:
    with gr.Blocks(title="Swift Crop Ratio Diagnostics") as app:
        with gr.Row(elem_classes="report-toolbar"):
            gr.Markdown("**Swift Crop Ratio Diagnostics**")
            gr.DownloadButton("Summary JSON", value=str(report_dir / "summary.json"), size="sm")
            gr.DownloadButton("Details CSV", value=str(report_dir / "details.csv"), size="sm")
            gr.DownloadButton("Details JSONL", value=str(report_dir / "details.jsonl"), size="sm")
        gr.HTML(
            '<div class="report-shell"><iframe src="/report/report.html" '
            'title="Swift crop ratio diagnostic report"></iframe></div>'
        )
    return app


def build_app(report_dir: Path, *, username: str | None = None, password: str | None = None) -> FastAPI:
    if bool(username) != bool(password):
        raise ValueError("username and password must be set together")
    report_dir = validate_report_dir(report_dir)
    root = FastAPI(title="Swift Crop Ratio Diagnostics")
    root.mount("/report", StaticFiles(directory=report_dir, html=True), name="report")
    auth = (username, password) if username and password else None
    return gr.mount_gradio_app(root, build_report_blocks(report_dir), path="/", auth=auth, css=REPORT_CSS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a static crop diagnostic report through Gradio.")
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7865)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.server_port <= 0:
        raise ValueError("server-port must be positive")
    app = build_app(
        args.report_dir,
        username=os.getenv("GRADIO_AUTH_USERNAME"),
        password=os.getenv("GRADIO_AUTH_PASSWORD"),
    )
    uvicorn.run(app, host=args.server_name, port=args.server_port)


if __name__ == "__main__":
    main()