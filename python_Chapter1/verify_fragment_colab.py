"""Validate executed GPU notebook, embedded video, benchmark and float32 fields."""
import argparse
import base64
import io
import json
from pathlib import Path
import re
import statistics
import zipfile

import numpy as np


def joined(value):
    return "".join(value) if isinstance(value, list) else value


def verify(notebook_path, archive_path):
    notebook = json.loads(notebook_path.read_text())
    outputs = [o for c in notebook["cells"] for o in c.get("outputs", [])]
    errors = [o for o in outputs if o.get("output_type") == "error"]
    if errors:
        raise AssertionError([(o["ename"], o["evalue"]) for o in errors])
    text = "".join(joined(o.get("text", "")) for o in outputs)
    assert "NVIDIA_EGL_OK" in text and "FRAGMENT_NOTEBOOK_OK" in text
    assert "Ran 4 tests" in text and "\nOK\n" in text
    video_cell = next(c for c in notebook["cells"] if c["id"] == "run-video")
    html = "\n".join(joined(o.get("data", {}).get("text/html", ""))
                     for o in video_cell.get("outputs", []))
    embedded = re.findall(r"data:video/mp4;base64,([A-Za-z0-9+/=\s]+)", html)
    assert len(embedded) == 1
    with zipfile.ZipFile(archive_path) as archive:
        report = json.loads(archive.read("verification.json"))
        benchmark = json.loads(archive.read("benchmark.json"))
        context = json.loads(archive.read("opengl_context.json"))
        assert "nvidia" in context["GL_VENDOR"].lower(), context
        assert not any(s in context["GL_RENDERER"].lower() for s in ("llvmpipe", "softpipe", "swiftshader"))
        assert context == report["opengl"] == benchmark["opengl"]
        assert report["tests_ok"] and report["tests_run"] == 4
        assert base64.b64decode(embedded[0]) == archive.read("fragment_tm.mp4")
        video = report["video"]
        assert report["ffprobe"]["codec_name"] == "h264"
        assert int(report["ffprobe"]["nb_read_frames"]) == video["frames"]
        assert benchmark["dtype"] == "float32"
        for row in benchmark["repeats"]:
            assert row["gpu_seconds"] > 0 and row["wall_seconds"] > 0
        with np.load(io.BytesIO(archive.read("fragment_tm.npz"))) as data:
            n = int(data["size"])
            assert n == video["size"] == benchmark["size"]
            assert data["ez"].shape == (n + 1, n + 1)
            assert data["hx"].shape == (n + 1, n)
            assert data["hy"].shape == (n, n + 1)
            assert int(data["step"]) == video["steps"]
            for key in ("ez", "ezx", "ezy", "hx", "hy"):
                assert data[key].dtype == np.dtype("float32")
                assert np.isfinite(data[key]).all()
            assert np.all(data["ez"][[0, -1]] == 0)
            assert np.all(data["ez"][:, [0, -1]] == 0)
    median_gpu = statistics.median(r["gpu_ms_per_step"] for r in benchmark["repeats"])
    median_wall = statistics.median(r["wall_ms_per_step"] for r in benchmark["repeats"])
    print(context)
    print(f"GPU tests: 4 passed; {n}x{n}, float32; embedded MP4 matches downloaded video")
    print(f"Video: {video['steps']} steps, {video['frames']} frames, {video['fps']} fps")
    print(f"Median GPU interval: {median_gpu:.4f} ms/step; wall: {median_wall:.4f} ms/step")
    print("Fragment notebook and artifacts verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    verify(args.notebook, args.archive)
