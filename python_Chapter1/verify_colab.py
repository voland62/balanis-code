"""Check CLI notebook outputs and compare downloaded Colab fields with NumPy locally."""
import argparse
import base64
import io
import json
from pathlib import Path
import re
import zipfile

import numpy as np

from fdtd import Simulation


def joined(value):
    return "".join(value) if isinstance(value, list) else value


def verify(notebook_path, archive_path):
    notebook = json.loads(notebook_path.read_text())
    outputs = [output for cell in notebook["cells"] for output in cell.get("outputs", [])]
    errors = [output for output in outputs if output.get("output_type") == "error"]
    if errors:
        raise AssertionError(f"Notebook errors: {[(e['ename'], e['evalue']) for e in errors]}")
    stream = "".join(joined(o.get("text", "")) for o in outputs)
    assert "FDTD_NOTEBOOK_OK" in stream, "Missing completion marker"
    assert "Ran 4 tests" in stream and "\nOK\n" in stream, "Missing successful numerical tests"
    html = "\n".join(joined(o.get("data", {}).get("text/html", "")) for o in outputs)
    videos = re.findall(r'data:video/mp4;base64,([A-Za-z0-9+/=\s]+)', html)
    with zipfile.ZipFile(archive_path) as archive:
        report = json.loads(archive.read("verification.json"))
        assert len(videos) == len(report), "Missing embedded videos"
        for row, video in zip(report, videos):
            name = row["scenario"]
            assert base64.b64decode(video) == archive.read(f"{name}.mp4"), name
            simulation = Simulation(name)
            for _ in range(row["steps"]):
                simulation.step()
            maximum_error = 0.0
            with np.load(io.BytesIO(archive.read(f"{name}.npz"))) as remote:
                assert int(remote["step"]) == simulation.n
                np.testing.assert_allclose(remote["dt"], simulation.dt, rtol=1e-14)
                for key, local in simulation.fields.items():
                    np.testing.assert_allclose(remote[key], local, rtol=1e-11, atol=1e-10,
                                               err_msg=f"{name}.{key}")
                    maximum_error = max(maximum_error, float(np.max(np.abs(remote[key] - local))))
            print(f"{name}: {row['frames']} frames; embedded MP4 matches; "
                  f"max |Colab - local| = {maximum_error:.3g}")
    print("Colab notebook outputs and numerical results verified")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    verify(args.notebook, args.archive)
