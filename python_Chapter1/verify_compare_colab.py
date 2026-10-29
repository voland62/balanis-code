"""Check executed comparison notebook and independently recompute its summaries."""
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
    assert not errors, [(o["ename"], o["evalue"]) for o in errors]
    text = "".join(joined(o.get("text", "")) for o in outputs)
    assert "COMPARE_NOTEBOOK_OK" in text and "Ran 9 tests" in text and "\nOK\n" in text
    with zipfile.ZipFile(archive_path) as archive:
        report = json.loads(archive.read("comparison.json"))
        assert report["tests_ok"] and report["tests_run"] == 9
        assert report["dtype"] == "float32" and report["workgroup_size"] == [16, 16, 1]
        assert "nvidia" in report["opengl"]["GL_VENDOR"].lower()
        assert report["opengl"] == json.loads(archive.read("opengl_context.json"))
        assert len(report["commit"]) == 40 and len(report["module_sha256"]) == 4
        assert report["repeats"] >= 2 and report["repeats"] % 2 == 0
        assert len(report["trials"]) == len(report["sizes"]) * report["repeats"] * 2
        assert len(report["field_checks"]) == len(report["sizes"]) * report["repeats"]
        for size, summary in zip(report["sizes"], report["summary"], strict=True):
            assert size == summary["size"]
            for repeat in range(report["repeats"]):
                pair = sorted((t for t in report["trials"] if t["size"] == size and t["repeat"] == repeat),
                              key=lambda t: t["position"])
                expected = ["fragment", "compute"] if repeat % 2 == 0 else ["compute", "fragment"]
                assert [t["backend"] for t in pair] == expected
                check = [r for r in report["field_checks"] if r["size"] == size and r["repeat"] == repeat]
                assert len(check) == 1 and 0 <= check[0]["max_relative_error"] < 5e-5
            for backend in ("fragment", "compute"):
                for metric in ("gpu", "wall"):
                    values = [t[f"{metric}_ms_per_step"] for t in report["trials"]
                              if t["size"] == size and t["backend"] == backend]
                    assert len(values) == report["repeats"]
                    assert all(np.isfinite(v) and v > 0 for v in values)
                    assert summary[f"{backend}_{metric}"] == statistics.median(values)
                    assert summary[f"{backend}_{metric}_min"] == min(values)
                    assert summary[f"{backend}_{metric}_max"] == max(values)
            for metric in ("gpu", "wall"):
                assert summary[f"speedup_{metric}"] == summary[f"fragment_{metric}"] / summary[f"compute_{metric}"]
            print(f"{size}: Fragment {summary['fragment_gpu']:.4f}, Compute {summary['compute_gpu']:.4f} ms/step; F/C {summary['speedup_gpu']:.3f}")
        assert archive.read("comparison.png").startswith(b"\x89PNG")
        assert b"<svg" in archive.read("comparison.svg")
        html = "".join(joined(o.get("data", {}).get("text/html", ""))
                       for c in notebook["cells"] if c["id"] == "videos" for o in c.get("outputs", []))
        embedded = re.findall(r"data:video/mp4;base64,([A-Za-z0-9+/=\s]+)", html)
        assert len(embedded) == len(report["videos"])
        fields = []
        for index, backend in enumerate(report["videos"]):
            assert base64.b64decode(embedded[index]) == archive.read(f"{backend}.mp4")
            with np.load(io.BytesIO(archive.read(f"{backend}.npz"))) as data:
                n = int(data["size"])
                assert n == report["videos"][backend]["size"]
                arrays = {k: data[k].copy() for k in ("ez", "ezx", "ezy", "hx", "hy")}
                assert arrays["ez"].shape == (n + 1, n + 1)
                assert all(a.dtype == np.float32 and np.isfinite(a).all() for a in arrays.values())
                fields.append(arrays)
        if len(fields) == 2:
            for key in fields[0]:
                np.testing.assert_allclose(fields[0][key], fields[1][key], rtol=5e-5, atol=1e-6)
    print(report["opengl"])
    print("Comparison notebook verified: tests, paired timings, field checks, summaries, plots and embedded videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    verify(args.notebook, args.archive)
