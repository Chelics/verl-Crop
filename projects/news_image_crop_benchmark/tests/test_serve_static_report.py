import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "serve_static_report.py"
    spec = importlib.util.spec_from_file_location("test_serve_static_report_module", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_report(report_dir: Path) -> None:
    report_dir.mkdir()
    (report_dir / "report.html").write_text("<h1>Report body</h1>", encoding="utf-8")
    (report_dir / "summary.json").write_text("{}", encoding="utf-8")
    (report_dir / "details.jsonl").write_text("{}\n", encoding="utf-8")
    (report_dir / "details.csv").write_text("image_id\nsample\n", encoding="utf-8")
    assets = report_dir / "assets"
    assets.mkdir()
    (assets / "sample.txt").write_text("asset", encoding="utf-8")


def test_serves_gradio_shell_report_and_assets(tmp_path):
    module = load_module()
    report_dir = tmp_path / "report"
    write_report(report_dir)

    app = module.build_app(report_dir)
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        report = client.get("/report/report.html")
        assert report.status_code == 200
        assert "Report body" in report.text
        asset = client.get("/report/assets/sample.txt")
        assert asset.status_code == 200
        assert asset.text == "asset"


def test_requires_complete_report_and_paired_auth(tmp_path):
    module = load_module()
    report_dir = tmp_path / "report"
    report_dir.mkdir()

    try:
        module.build_app(report_dir)
    except FileNotFoundError as error:
        assert "report.html" in str(error)
    else:
        raise AssertionError("incomplete report should fail")

    write_report(tmp_path / "complete")
    try:
        module.build_app(tmp_path / "complete", username="user")
    except ValueError as error:
        assert "set together" in str(error)
    else:
        raise AssertionError("unpaired auth should fail")