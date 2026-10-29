"""Float32 TMz FDTD using OpenGL fragment shaders (offscreen EGL).

The same definitions are placed directly in fdtd_fragment_colab.ipynb.
Arrays use [i, j] = [row, column]; OpenGL texel (x, y) corresponds to (j, i).
"""
from pathlib import Path
import ctypes.util
import json
import math
import os
import time

import numpy as np


# %% EGL

def create_nvidia_context(directory):
    """Select the NVIDIA EGL vendor before the first EGL context is created."""
    import moderngl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    candidates = [Path("/usr/lib64-nvidia/libEGL_nvidia.so.0"),
                  Path("/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0"),
                  Path("/usr/local/nvidia/lib64/libEGL_nvidia.so.0")]
    library = next((str(p) for p in candidates if p.exists()), None)
    library = library or ctypes.util.find_library("EGL_nvidia")
    if library:
        vendor = directory / "nvidia_egl_vendor.json"
        vendor.write_text(json.dumps({"file_format_version": "1.0.0",
                                      "ICD": {"library_path": library}}))
        os.environ["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(vendor.resolve())
    try:
        ctx = moderngl.create_context(standalone=True, backend="egl", require=430)
    except Exception as error:
        raise RuntimeError("NVIDIA EGL/OpenGL 4.3 недоступен. Проверьте GPU-среду; "
                           "после смены EGL vendor перезапустите ядро.") from error
    info = {key: ctx.info[key] for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}
    if "nvidia" not in info["GL_VENDOR"].lower() or any(
            name in info["GL_RENDERER"].lower() for name in ("llvmpipe", "softpipe", "swiftshader")):
        ctx.release()
        raise RuntimeError(f"Требуется аппаратный NVIDIA OpenGL, получен {info}. "
                           "Выберите GPU и перезапустите ядро.")
    (directory / "opengl_context.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    return ctx


# %% model
class TMConfig:
    """Fixed 0.15 m square; refining the grid also reduces dt and scales the pulse."""

    def __init__(self, size=1000, boundary="pml"):
        if not isinstance(size, int) or isinstance(size, bool) or size < 16:
            raise ValueError("size must be an integer >= 16")
        if boundary not in ("pml", "pec"):
            raise ValueError("boundary must be pml or pec")
        self.size = size
        self.boundary = boundary
        self.pml = max(2, round(6 * size / 50)) if boundary == "pml" else 0
        self.mu = 4 * np.pi * 1e-7
        self.eps = 8.854e-12
        self.c = 1 / np.sqrt(self.mu * self.eps)
        self.dx = 0.15 / size
        self.dt = self.dx / (2 * self.c)
        self.pulse_width = 25 * size / 50
        self.default_steps = math.ceil((111 if self.pml else 210) * size / 50)
        self.center = size // 2

    def source(self, step):
        u = 4 * (step / self.pulse_width - 1)
        return np.float32(2 * u * math.exp(-u * u) / self.dx)

    def coefficients(self):
        n = self.size
        # qe = (ax, bx, ay, by), qh = (aHx, bHx, aHy, bHy).
        qe = np.empty((n + 1, n + 1, 4), dtype=np.float32)
        qh = np.empty_like(qe)
        qe[..., (0, 2)] = qh[..., (0, 2)] = 1
        qe[..., (1, 3)] = self.dt / self.eps / self.dx
        qh[..., (1, 3)] = self.dt / self.mu / self.dx
        p = self.pml
        if p:
            m = np.arange(1, p + 1, dtype=np.float64)
            sigma_max = -3 * self.eps * self.c * np.log(1e-5) / (2 * self.dx * p)
            sigma = sigma_max * ((m - .5) / (p + .5))**2
            rho = sigma_max * (self.mu / self.eps) * (m / (p + .5))**2
            ca = np.exp(-sigma * self.dt / self.eps)
            cb = -(ca - 1) / sigma / self.dx
            da = np.exp(-rho * self.dt / self.mu)
            db = -(da - 1) / rho / self.dx
            # Match the placement of the split-field Berenger PML in TM_OPEN.m.
            for array, axis, channel, a, b, offset in (
                    (qe, 0, 0, ca, cb, 1), (qe, 1, 2, ca, cb, 1),
                    (qh, 1, 0, da, db, 0), (qh, 0, 2, da, db, 0)):
                for k in range(p):
                    for position, coefficient in ((offset + k, p - 1 - k), (n - p + k, k)):
                        index = [slice(1, n), slice(1, n)]
                        index[axis] = position
                        array[(*index, channel)] = a[coefficient]
                        array[(*index, channel + 1)] = b[coefficient]
        return qe, qh


class TMReference:
    """Small NumPy float32 oracle; used for tests, never for the GPU benchmark."""

    def __init__(self, config):
        self.config = config
        n = config.size
        self.e = np.zeros((n + 1, n + 1, 4), dtype=np.float32)
        self.h = np.zeros((n + 1, n + 1, 2), dtype=np.float32)
        self.qe, self.qh = config.coefficients()
        self.n = 0

    def step(self):
        c = self.config
        n = c.size
        self.n += 1
        region = np.s_[1:n, 1:n]
        xd = self.h[1:n, 1:n, 1] - self.h[:n - 1, 1:n, 1]
        yd = self.h[1:n, :n - 1, 0] - self.h[1:n, 1:n, 0]
        if c.pml:
            self.e[region + (1,)] = (self.qe[region + (0,)] * self.e[region + (1,)]
                                    + self.qe[region + (1,)] * xd)
            self.e[region + (2,)] = (self.qe[region + (2,)] * self.e[region + (2,)]
                                    + self.qe[region + (3,)] * yd)
            self.e[region + (0,)] = self.e[region + (1,)] + self.e[region + (2,)]
        else:
            self.e[region + (0,)] += self.qe[region + (1,)] * (xd + yd)
        self.e[c.center, c.center, 0] = c.source(self.n)
        self.h[:, :n, 0] = (self.qh[:, :n, 0] * self.h[:, :n, 0]
                           + self.qh[:, :n, 1] * (self.e[:, :n, 0] - self.e[:, 1:, 0]))
        self.h[:n, :, 1] = (self.qh[:n, :, 2] * self.h[:n, :, 1]
                           + self.qh[:n, :, 3] * (self.e[1:, :, 0] - self.e[:n, :, 0]))


# %% shaders
VERTEX_SHADER = """#version 430
// One full-screen triangle; no vertex buffer and no diagonal between triangles.
void main() {
    vec2 positions[3] = vec2[](vec2(-1,-1), vec2(3,-1), vec2(-1,3));
    gl_Position = vec4(positions[gl_VertexID], 0, 1);
}
"""

E_SHADER = """#version 430
uniform sampler2D old_e;
uniform sampler2D old_h;
uniform sampler2D qe;
uniform int size;
uniform bool use_pml;
uniform float source;
layout(location=0) out vec4 new_e;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy); // (j, i)
    if (p.x == 0 || p.y == 0 || p.x == size || p.y == size) {
        new_e = vec4(0); return;
    }
    vec4 e = texelFetch(old_e, p, 0);
    vec4 q = texelFetch(qe, p, 0);
    vec2 h = texelFetch(old_h, p, 0).rg;
    float xd = h.y - texelFetch(old_h, p - ivec2(0,1), 0).g;
    float yd = texelFetch(old_h, p - ivec2(1,0), 0).r - h.x;
    // Separate operations also make this an interpretable FP32 baseline.
    precise float ex = q.x * e.y + q.y * xd;
    precise float ey = q.z * e.z + q.w * yd;
    precise float ez = use_pml ? ex + ey : e.x + q.y * (xd + yd);
    if (all(equal(p, ivec2(size / 2)))) ez = source;
    new_e = vec4(ez, use_pml ? ex : 0, use_pml ? ey : 0, 0);
}
"""

H_SHADER = """#version 430
uniform sampler2D new_e;
uniform sampler2D old_h;
uniform sampler2D qh;
uniform int size;
layout(location=0) out vec2 new_h;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 h = texelFetch(old_h, p, 0).rg;
    vec4 q = texelFetch(qh, p, 0);
    float ez = texelFetch(new_e, p, 0).r;
    precise float hx = 0;
    precise float hy = 0;
    if (p.x < size)
        hx = q.x * h.x + q.y * (ez - texelFetch(new_e, p + ivec2(1,0), 0).r);
    if (p.y < size)
        hy = q.z * h.y + q.w * (texelFetch(new_e, p + ivec2(0,1), 0).r - ez);
    new_h = vec2(hx, hy);
}
"""

COLOR_SHADER = """#version 430
uniform sampler2D field;
uniform float limit;
uniform vec2 output_size;
layout(location=0) out vec4 color;
void main() {
    // Blue = negative, white = zero, red = positive. GL row 0 is the lower row.
    float value = texture(field, gl_FragCoord.xy / output_size).r;
    float t = clamp(value / limit, -1.0, 1.0);
    vec3 rgb = t < 0 ? mix(vec3(1), vec3(0.1,0.25,0.9), -t)
                     : mix(vec3(1), vec3(0.9,0.15,0.1), t);
    color = vec4(rgb, 1);
}
"""


# %% gpu
class FragmentTM:
    """Two raster passes per FDTD step; ping-pong RGBA32F / RG32F textures."""

    def __init__(self, ctx, config):
        import moderngl
        self.ctx, self.config = ctx, config
        self.front, self.n = 0, 0
        n = config.size + 1
        if n > ctx.info["GL_MAX_TEXTURE_SIZE"]:
            raise ValueError("Grid exceeds GL_MAX_TEXTURE_SIZE")
        self.resources = []
        try:
            def texture(channels, values=None):
                if values is None:
                    values = np.zeros((n, n, channels), dtype=np.float32)
                tex = ctx.texture((n, n), channels, np.ascontiguousarray(values).tobytes(), dtype="f4")
                self.resources.append(tex)
                tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
                tex.repeat_x = tex.repeat_y = False
                return tex

            qe, qh = config.coefficients()
            self.qe, self.qh = texture(4, qe), texture(4, qh)
            self.e = [texture(4), texture(4)]
            self.h = [texture(2), texture(2)]
            self.efbo = [ctx.framebuffer([tex]) for tex in self.e]
            self.hfbo = [ctx.framebuffer([tex]) for tex in self.h]
            self.resources += self.efbo + self.hfbo
            self.eprog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=E_SHADER)
            self.resources.append(self.eprog)
            self.hprog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=H_SHADER)
            self.resources.append(self.hprog)
            self.evao = ctx.vertex_array(self.eprog, [])
            self.hvao = ctx.vertex_array(self.hprog, [])
            self.resources += [self.evao, self.hvao]
            self.eprog["old_e"].value, self.eprog["old_h"].value, self.eprog["qe"].value = 0, 1, 2
            self.hprog["new_e"].value, self.hprog["old_h"].value, self.hprog["qh"].value = 0, 1, 2
            self.eprog["size"].value = self.hprog["size"].value = config.size
            self.eprog["use_pml"].value = bool(config.pml)
            ctx.disable(moderngl.DEPTH_TEST | moderngl.BLEND | moderngl.CULL_FACE)
        except BaseException:
            self.release()
            raise

    def step(self):
        self.n += 1
        back = 1 - self.front
        self.e[self.front].use(0)
        self.h[self.front].use(1)
        self.qe.use(2)
        self.eprog["source"].value = float(self.config.source(self.n))
        self.efbo[back].use()
        self.evao.render(vertices=3)
        # New E is complete before H reads it; neither pass reads its own target.
        self.e[back].use(0)
        self.qh.use(2)
        self.hfbo[back].use()
        self.hvao.render(vertices=3)
        self.front = back

    def read_fields(self):
        """Explicit GPU -> CPU transfer. Never called inside timing loops."""
        n = self.config.size + 1
        e = np.frombuffer(self.e[self.front].read(), np.float32).reshape(n, n, 4).copy()
        h = np.frombuffer(self.h[self.front].read(), np.float32).reshape(n, n, 2).copy()
        return e, h

    def reset(self):
        n = self.config.size + 1
        for tex in self.e:
            tex.write(bytes(n * n * 4 * 4))
        for tex in self.h:
            tex.write(bytes(n * n * 2 * 4))
        self.front, self.n = 0, 0

    def release(self):
        for resource in reversed(self.resources):
            resource.release()
        self.resources.clear()


# %% benchmark

def benchmark_fragment(ctx, config, steps=300, repeats=3, warmup=300):
    """GPU timer query + synchronized wall time, excluding allocation and readback."""
    if steps < 1 or repeats < 1 or warmup < 1:
        raise ValueError("steps, repeats and warmup must be positive")
    sim = FragmentTM(ctx, config)
    results = []
    try:
        for _ in range(warmup):
            sim.step()
        ctx.finish()
        for repeat in range(repeats):
            sim.reset()
            ctx.finish()
            query = ctx.query(time=True)
            start = time.perf_counter()
            with query:
                for _ in range(steps):
                    sim.step()
            ctx.finish()
            wall_seconds = time.perf_counter() - start
            gpu_seconds = query.elapsed / 1e9
            if gpu_seconds <= 0:
                raise RuntimeError("GPU timer returned no valid duration")
            results.append(dict(repeat=repeat + 1, gpu_seconds=gpu_seconds,
                                wall_seconds=wall_seconds,
                                gpu_ms_per_step=1000 * gpu_seconds / steps,
                                wall_ms_per_step=1000 * wall_seconds / steps,
                                million_cell_steps_per_second=config.size**2 * steps / gpu_seconds / 1e6))
        return dict(size=config.size, dtype="float32", boundary=config.boundary,
                    steps=steps, warmup=warmup, repeats=results,
                    note="GPU query spans the submitted passes and may include GPU idle gaps "
                         "while Python submits commands; wall time includes host dispatch.")
    finally:
        sim.release()


# %% video

def render_fragment_video(ctx, config, output, steps, stride=20, fps=25, resolution=800, limit=50):
    """GPU color pass -> RGB readback -> FFmpeg; independent of the solver benchmark."""
    import moderngl
    import subprocess
    if steps < stride or stride < 1 or fps < 1 or resolution < 2 or resolution % 2 or limit <= 0:
        raise ValueError("Invalid video settings; resolution must be even")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sim = FragmentTM(ctx, config)
    resources = []
    process = None
    frame_count = 0
    start = time.perf_counter()
    try:
        rgb = ctx.texture((resolution, resolution), 4, dtype="f1")
        resources.append(rgb)
        fbo = ctx.framebuffer([rgb])
        resources.append(fbo)
        program = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=COLOR_SHADER)
        resources.append(program)
        vao = ctx.vertex_array(program, [])
        resources.append(vao)
        program["field"].value = 0
        program["limit"].value = float(limit)
        program["output_size"].value = (resolution, resolution)
        # Raw OpenGL rows are bottom-up; vflip restores the displayed orientation.
        command = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
                   "-video_size", f"{resolution}x{resolution}", "-framerate", str(fps),
                   "-i", "pipe:0", "-vf", "vflip", "-an", "-c:v", "libx264", "-preset", "fast",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
        # Log to disk rather than a stderr pipe that can block during encoding.
        with output.with_suffix(".ffmpeg.log").open("wb") as log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            try:
                for _ in range(steps):
                    sim.step()
                    if sim.n % stride == 0:
                        fbo.use()
                        sim.e[sim.front].use(0)
                        vao.render(vertices=3)
                        process.stdin.write(fbo.read(components=3, alignment=1))
                        frame_count += 1
            finally:
                process.stdin.close()
                returncode = process.wait()
            if returncode:
                raise RuntimeError(f"FFmpeg failed ({returncode}); see {output.with_suffix('.ffmpeg.log')}")
        e, h = sim.read_fields()
        np.savez_compressed(output.with_suffix(".npz"), ez=e[..., 0], ezx=e[..., 1], ezy=e[..., 2],
                            hx=h[:, :-1, 0], hy=h[:-1, :, 1], step=sim.n,
                            dx=config.dx, dt=config.dt, size=config.size, boundary=config.boundary)
        return dict(frames=frame_count, fps=fps, steps=steps, size=config.size,
                    resolution=resolution, wall_seconds=time.perf_counter() - start,
                    maximum_field=float(np.max(np.abs(e[..., 0]))), video_bytes=output.stat().st_size)
    finally:
        for resource in reversed(resources):
            resource.release()
        sim.release()
