"""NumPy port of W. V. Andrew's 1995 Chapter 1 MATLAB FDTD examples.

Array axes and update ordering follow the MATLAB originals. Indices in this
module are zero based; field components retain their staggered Yee shapes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


class Simulation:
    """Two-dimensional vacuum TMz or TEz simulation with original defaults."""

    def __init__(self, scenario: str):
        if scenario not in ("tm_box", "tm_open", "te_horn"):
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario = scenario
        self.te = scenario == "te_horn"
        self.size = s = 100 if self.te else 50
        self.default_steps = 210 if scenario != "tm_open" else 111
        self.npml = 8 if self.te else (6 if scenario == "tm_open" else 0)
        self.mu = 4 * np.pi * 1e-7
        self.eps = 8.854e-12
        self.c = 1 / np.sqrt(self.mu * self.eps)
        self.impedance = np.sqrt(self.mu / self.eps)
        self.frequency = 9.84252e9
        self.dx = self.c / self.frequency / 12 if self.te else 0.003
        self.dt = self.dx / self.c / 2
        self.n = 0
        if self.te:
            shapes = {"ex": (s, s + 1), "ey": (s + 1, s),
                      "hz": (s, s), "hzx": (s, s), "hzy": (s, s)}
            self.ic, self.jc = s // 2 - 21, s // 2 - 1
            components = ("ex", "ey", "hzx", "hzy")
        else:
            shapes = {"ez": (s + 1, s + 1), "ezx": (s + 1, s + 1),
                      "ezy": (s + 1, s + 1), "hx": (s + 1, s), "hy": (s, s + 1)}
            self.ic = self.jc = s // 2
            components = ("ezx", "ezy", "hx", "hy")
        self.fields = {name: np.zeros(shape) for name, shape in shapes.items()}
        self.coefficients = {}
        for name in components:
            constant = self.eps if name.startswith("e") else self.mu
            self.coefficients[name] = (np.ones(shapes[name]),
                                       np.full(shapes[name], self.dt / constant / self.dx))
        if self.npml:
            self._pml()
        if self.te:
            self._horn()

    def _pml(self):
        p, s = self.npml, self.size
        m = np.arange(1, p + 1)
        sigmax = -3 * self.eps * self.c * np.log(1e-5) / (2 * self.dx * p)
        sigma = sigmax * ((m - 0.5) / (p + 0.5)) ** 2
        rho = sigmax * self.impedance**2 * (m / (p + 0.5)) ** 2
        ca = np.exp(-sigma * self.dt / self.eps)
        cb = -(np.exp(-sigma * self.dt / self.eps) - 1) / sigma / self.dx
        da = np.exp(-rho * self.dt / self.mu)
        db = -(np.exp(-rho * self.dt / self.mu) - 1) / rho / self.dx
        # Each tuple gives the array axis normal to the absorbing boundary.
        electric = (("ey", 0), ("ex", 1)) if self.te else (("ezx", 0), ("ezy", 1))
        magnetic = (("hzx", 0), ("hzy", 1)) if self.te else (("hy", 0), ("hx", 1))
        for components, a, b, offset in ((electric, ca, cb, 1), (magnetic, da, db, 0)):
            for name, axis in components:
                aa, bb = self.coefficients[name]
                for k in range(p):
                    for position, coefficient_index in ((offset + k, p - 1 - k), (s - p + k, k)):
                        index = [slice(1, s), slice(1, s)]
                        index[axis] = position
                        aa[tuple(index)] = a[coefficient_index]
                        bb[tuple(index)] = b[coefficient_index]

    def _horn(self):
        i, j = self.ic, self.jc
        aex, bex = self.coefficients["ex"]
        aey, bey = self.coefficients["ey"]
        for wall in (j - 2, j + 2):
            aex[i - 13:i + 3, wall] = -1
            bex[i - 13:i + 3, wall] = 0
        aey[i - 13, j - 2:j + 2] = -1
        bey[i - 13, j - 2:j + 2] = 0
        for offset in range(3, 12):
            width = (offset + 3) // 2
            for wall in (j - width, j + width):
                aex[i + offset, wall] = -1
                bex[i + offset, wall] = 0
            if offset % 2:
                for wall in (j - width, j + width - 1):
                    aey[i + offset, wall] = -1
                    bey[i + offset, wall] = 0

    @property
    def field(self):
        """Signed field displayed by the MATLAB example (Ey or Ez)."""
        return self.fields["ey" if self.te else "ez"]

    def step(self):
        self.n += 1
        f, q, s = self.fields, self.coefficients, self.size
        if self.te:
            a, b = q["ex"]
            f["ex"][:, 1:s] = a[:, 1:s] * f["ex"][:, 1:s] + b[:, 1:s] * (f["hz"][:, 1:] - f["hz"][:, :-1])
            a, b = q["ey"]
            f["ey"][1:s, :] = a[1:s, :] * f["ey"][1:s, :] + b[1:s, :] * (f["hz"][:-1, :] - f["hz"][1:, :])
            f["ey"][self.ic - 10, self.jc - 2:self.jc + 2] = (
                (1 - np.exp(-(self.n / 20)**2)) * self.impedance
                * np.sin(2 * np.pi * self.frequency * self.n * self.dt))
            a, b = q["hzx"]
            f["hzx"][:] = a * f["hzx"] + b * (f["ey"][:-1, :] - f["ey"][1:, :])
            a, b = q["hzy"]
            f["hzy"][:] = a * f["hzy"] + b * (f["ex"][:, 1:] - f["ex"][:, :-1])
            f["hz"][:] = f["hzx"] + f["hzy"]
        else:
            xdiff = f["hy"][1:, 1:s] - f["hy"][:-1, 1:s]
            ydiff = f["hx"][1:s, :-1] - f["hx"][1:s, 1:]
            interior = np.s_[1:s, 1:s]
            if self.npml:
                for name, difference in (("ezx", xdiff), ("ezy", ydiff)):
                    a, b = q[name]
                    f[name][interior] = a[interior] * f[name][interior] + b[interior] * difference
                f["ez"][interior] = f["ezx"][interior] + f["ezy"][interior]
            else:
                f["ez"][interior] += self.dt / self.eps / self.dx * (xdiff + ydiff)
            dum = 4 / 25 / self.dt * (self.n * self.dt - 25 * self.dt)
            f["ez"][self.ic, self.jc] = 2 * dum * np.exp(-dum**2) / self.dx
            a, b = q["hx"]
            f["hx"][:] = a * f["hx"] + b * (f["ez"][:, :-1] - f["ez"][:, 1:])
            a, b = q["hy"]
            f["hy"][:] = a * f["hy"] + b * (f["ez"][1:, :] - f["ez"][:-1, :])
        return self.field

    def display_field(self):
        if self.te:
            # Preserve abs(Ey + epsilon), rather than silently changing the original.
            with np.errstate(divide="ignore"):
                return np.log10(np.abs(self.field + 1e-6))
        return self.field

    def save(self, path):
        """Save final staggered fields and metadata for numerical comparisons."""
        np.savez_compressed(path, **self.fields, step=self.n, dx=self.dx, dt=self.dt,
                            scenario=self.scenario, npml=self.npml)


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", choices=("tm_box", "tm_open", "te_horn"))
    parser.add_argument("--steps", type=positive_int, help="default: original MATLAB duration")
    parser.add_argument("--stride", type=positive_int, default=3, help="steps per frame (default: 3)")
    parser.add_argument("--fps", type=positive_int, default=7)
    parser.add_argument("--output", type=Path, help="MP4 path; default: <scenario>.mp4")
    parser.add_argument("--no-video", action="store_true", help="run the solver without rendering")
    parser.add_argument("--data", type=Path, help="save final fields to NPZ")
    parser.add_argument("--snapshot", type=Path, help="save final visualization, e.g. PNG")
    args = parser.parse_args()
    sim = Simulation(args.scenario)
    steps = args.steps or sim.default_steps
    if not args.no_video and steps < args.stride:
        parser.error("steps must be at least stride to produce a video frame")
    if args.output and args.no_video:
        parser.error("--output cannot be used with --no-video")
    render = not args.no_video or args.snapshot is not None
    if render:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
        if not args.no_video and not FFMpegWriter.isAvailable():
            parser.error("ffmpeg is unavailable; enter the Nix development shell")
        fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
        limits = (-6, 3) if sim.te else (-50, 50)
        # MATLAB pcolor treats columns as horizontal, rows as vertical. No transpose.
        rows, cols = sim.field.shape
        mesh = ax.pcolormesh(np.arange(1, cols + 1), np.arange(1, rows + 1),
                             sim.display_field(), shading="gouraud", cmap="jet",
                             vmin=limits[0], vmax=limits[1])
        ax.set(xlim=(0, sim.size), ylim=(0, sim.size), aspect="equal",
               xlabel="j (cell index, MATLAB convention)", ylabel="i (cell index, MATLAB convention)")
        fig.colorbar(mesh, ax=ax, label="log10(|Ey + 10⁻⁶|)" if sim.te else "Ez (V/m)")
        title = ax.set_title("")

        def update_plot():
            mesh.set_array(sim.display_field())
            title.set_text(f"{sim.scenario.upper()} · step {sim.n} · t = {sim.n * sim.dt * 1e9:.3f} ns")

    frames = 0
    if args.no_video:
        for _ in range(steps):
            sim.step()
    else:
        output = args.output or Path(f"{args.scenario}.mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = FFMpegWriter(fps=args.fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        try:
            with writer.saving(fig, str(output), dpi=100):
                for _ in range(steps):
                    sim.step()
                    if sim.n % args.stride == 0:
                        update_plot()
                        writer.grab_frame()
                        frames += 1
        except BaseException:
            plt.close(fig)
            raise
        print(f"Video: {output} ({frames} frames, {args.fps} fps)")
    if args.data:
        args.data.parent.mkdir(parents=True, exist_ok=True)
        sim.save(args.data)
    if args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        update_plot()
        fig.savefig(args.snapshot, dpi=150)
    if render:
        plt.close(fig)
    print(f"{sim.scenario}: {sim.n} steps, dx={sim.dx:.8g} m, dt={sim.dt:.8g} s, "
          f"max |field|={np.max(np.abs(sim.field)):.8g}")


if __name__ == "__main__":
    main()
