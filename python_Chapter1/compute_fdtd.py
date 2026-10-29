"""Float32 TMz FDTD using OpenGL compute shaders and the fragment baseline's layout.

The educational notebook embeds these definitions and the common model directly.
For ordinary imports, fragment_fdtd supplies EGL, model and visualization shaders.
"""
from pathlib import Path
import time

import numpy as np

from fragment_fdtd import (
    COLOR_SHADER, VERTEX_SHADER, TMConfig, TMReference, create_nvidia_context,
)


# %% shaders
E_COMPUTE_SHADER = """#version 430
layout(local_size_x=16, local_size_y=16, local_size_z=1) in;
uniform sampler2D old_e;
uniform sampler2D old_h;
uniform sampler2D qe;
uniform int size;
uniform bool use_pml;
uniform float source;
layout(rgba32f, binding=0) writeonly uniform image2D output_e;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy); // (j, i), not a screen coordinate
    if (p.x > size || p.y > size) return; // Last workgroups can exceed the grid.
    if (p.x == 0 || p.y == 0 || p.x == size || p.y == size) {
        imageStore(output_e, p, vec4(0)); return;
    }
    vec4 e = texelFetch(old_e, p, 0);
    vec4 q = texelFetch(qe, p, 0);
    vec2 h = texelFetch(old_h, p, 0).rg;
    float xd = h.y - texelFetch(old_h, p - ivec2(0,1), 0).g;
    float yd = texelFetch(old_h, p - ivec2(1,0), 0).r - h.x;
    // Same arithmetic order and precise qualifiers as the fragment baseline.
    precise float ex = q.x * e.y + q.y * xd;
    precise float ey = q.z * e.z + q.w * yd;
    precise float ez = use_pml ? ex + ey : e.x + q.y * (xd + yd);
    if (all(equal(p, ivec2(size / 2)))) ez = source;
    imageStore(output_e, p, vec4(ez, use_pml ? ex : 0, use_pml ? ey : 0, 0));
}
"""

H_COMPUTE_SHADER = """#version 430
layout(local_size_x=16, local_size_y=16, local_size_z=1) in;
uniform sampler2D new_e;
uniform sampler2D old_h;
uniform sampler2D qh;
uniform int size;
layout(rg32f, binding=0) writeonly uniform image2D output_h;
void main() {
    ivec2 p = ivec2(gl_GlobalInvocationID.xy);
    if (p.x > size || p.y > size) return;
    vec2 h = texelFetch(old_h, p, 0).rg;
    vec4 q = texelFetch(qh, p, 0);
    float ez = texelFetch(new_e, p, 0).r;
    precise float hx = 0;
    precise float hy = 0;
    if (p.x < size)
        hx = q.x * h.x + q.y * (ez - texelFetch(new_e, p + ivec2(1,0), 0).r);
    if (p.y < size)
        hy = q.z * h.y + q.w * (texelFetch(new_e, p + ivec2(0,1), 0).r - ez);
    imageStore(output_h, p, vec4(hx, hy, 0, 0));
}
"""


# %% gpu
class ComputeTM:
    """Two compute dispatches per step; sampler reads and imageStore writes.

    16x16 workgroups, no shared memory and no fused temporal steps. All fields
    and coefficients use the same textures as FragmentTM, for a comparable base.
    """

    workgroup_size = (16, 16, 1)

    def __init__(self, ctx, config):
        import moderngl
        self.ctx, self.config = ctx, config
        self.front, self.n = 0, 0
        n = config.size + 1
        if n > ctx.info["GL_MAX_TEXTURE_SIZE"]:
            raise ValueError("Grid exceeds GL_MAX_TEXTURE_SIZE")
        self.groups = ((n + 15) // 16, (n + 15) // 16, 1)
        self.shader_barriers = (moderngl.TEXTURE_FETCH_BARRIER_BIT
                                | moderngl.SHADER_IMAGE_ACCESS_BARRIER_BIT)
        self.texture_update_barrier = moderngl.TEXTURE_UPDATE_BARRIER_BIT
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
            self.eprog = ctx.compute_shader(E_COMPUTE_SHADER)
            self.resources.append(self.eprog)
            self.hprog = ctx.compute_shader(H_COMPUTE_SHADER)
            self.resources.append(self.hprog)
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
        self.e[back].bind_to_image(0, read=False, write=True)
        self.eprog["source"].value = float(self.config.source(self.n))
        self.eprog.run(*self.groups)
        # imageStore is not automatically visible to later sampler reads.
        self.ctx.memory_barrier(self.shader_barriers)
        self.e[back].use(0)
        self.qh.use(2)
        self.h[back].bind_to_image(0, read=False, write=True)
        self.hprog.run(*self.groups)
        # Publish H to the next E pass, and E to the visualization sampler.
        # Also order later image writes after earlier shader memory accesses.
        self.ctx.memory_barrier(self.shader_barriers)
        self.front = back

    def read_fields(self):
        """Make image writes visible to glGetTexImage before GPU -> CPU transfer."""
        self.ctx.memory_barrier(self.texture_update_barrier)
        n = self.config.size + 1
        e = np.frombuffer(self.e[self.front].read(), np.float32).reshape(n, n, 4).copy()
        h = np.frombuffer(self.h[self.front].read(), np.float32).reshape(n, n, 2).copy()
        return e, h

    def reset(self):
        self.ctx.memory_barrier(self.texture_update_barrier)
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

def benchmark_compute(ctx, config, steps=300, repeats=3, warmup=300):
    """GPU timer query + synchronized wall time, excluding allocation and readback."""
    if steps < 1 or repeats < 1 or warmup < 1:
        raise ValueError("steps, repeats and warmup must be positive")
    sim = ComputeTM(ctx, config)
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
        return dict(backend="compute", workgroup_size=list(sim.workgroup_size),
                    size=config.size, dtype="float32", boundary=config.boundary,
                    steps=steps, warmup=warmup, repeats=results,
                    note="GPU query spans the submitted passes and may include GPU idle gaps "
                         "while Python submits commands; wall time includes host dispatch.")
    finally:
        sim.release()


# %% video

def render_compute_video(ctx, config, output, steps, stride=20, fps=25, resolution=800, limit=50):
    """GPU color pass -> RGB readback -> FFmpeg; independent of the solver benchmark."""
    import moderngl
    import subprocess
    if steps < stride or stride < 1 or fps < 1 or resolution < 2 or resolution % 2 or limit <= 0:
        raise ValueError("Invalid video settings; resolution must be even")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sim = ComputeTM(ctx, config)
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
