"""Generate the comparison notebook; solver imports are pinned to a published commit."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent
COMMIT = "9774c6155a3c2934c19a653a86f3f85062aba105"
MODULES = {
    name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in ("fragment_fdtd.py", "compute_fdtd.py", "fragment_checks.py", "compute_checks.py")
}


def build_notebook():
    cells = []

    def add(kind, ident, source):
        cell = dict(cell_type=kind, id=ident, metadata={},
                    source=textwrap.dedent(source).strip().splitlines(keepends=True))
        cell["source"][-1] += "\n"
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "intro", """
    # Fragment Shader vs Compute Shader: один GPU, одна модель

    Выберите **GPU-среду Colab**, затем выполните все ячейки.
    Сравниваем TMz FDTD (Ez/Hx/Hy), float32, PML/PEC, одинаковые коэффициенты,
    текстуры и два прохода E → H. Compute использует группы 16×16, без shared memory.
    Это сравнение двух конкретных реализаций, а не предел возможностей двух API.

    Решатели и их тесты импортируются из четырёх небольших `.py` файлов GitHub
    по закреплённому коммиту с проверкой SHA256. Учебные блокноты остаются
    самостоятельными. Здесь цикл эксперимента и анализ видны прямо в ячейках.
    """)
    add("code", "packages", "%pip -q install moderngl==5.12.0 glcontext==3.0.0")
    add("code", "download", f'''
    from pathlib import Path
    import hashlib
    import importlib
    import json
    import sys
    import urllib.request

    COMMIT = {COMMIT!r}
    MODULE_HASHES = {MODULES!r}
    BASE = f"https://raw.githubusercontent.com/voland62/balanis-code/{{COMMIT}}/python_Chapter1"
    OUTPUT = Path("/content/balanis_compare")
    MODULE_DIR = OUTPUT / "modules"
    MODULE_DIR.mkdir(parents=True, exist_ok=True)
    for name, expected in MODULE_HASHES.items():
        with urllib.request.urlopen(f"{{BASE}}/{{name}}", timeout=60) as response:
            data = response.read()
        assert hashlib.sha256(data).hexdigest() == expected, f"SHA256 mismatch: {{name}}"
        (MODULE_DIR / name).write_bytes(data)
    sys.path.insert(0, str(MODULE_DIR))
    # Повторный запуск ячейки должен импортировать именно проверенные исходники.
    for name in ("compute_checks", "fragment_checks", "compute_fdtd", "fragment_fdtd"):
        sys.modules.pop(name, None)
    importlib.invalidate_caches()
    print("Исходники:", COMMIT)
    print("SHA256 проверен:", ", ".join(MODULE_HASHES))
    ''')
    add("code", "context-tests", '''
    import platform
    import time
    import unittest
    import zipfile
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    import moderngl
    from IPython.display import display, HTML, FileLink, Video
    from fragment_fdtd import FragmentTM, TMConfig, create_nvidia_context, render_fragment_video
    from compute_fdtd import ComputeTM, render_compute_video
    from fragment_checks import FragmentTests
    from compute_checks import ComputeTests

    # Единственный контекст для обоих решателей; CPU/Mesa не допускается.
    ctx = create_nvidia_context(OUTPUT)
    FragmentTests.context = ComputeTests.context = ctx
    suite = unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromTestCase(FragmentTests),
        unittest.defaultTestLoader.loadTestsFromTestCase(ComputeTests),
    ])
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert test_result.wasSuccessful() and test_result.testsRun == 9 and not test_result.skipped
    print("NVIDIA_EGL_OK; 9 GPU tests passed")
    ''')
    add("markdown", "method", '''
    ## Методика и параметры

    Для каждого размера оба решателя создаются заранее в одном контексте.
    Перед каждой серией — сброс и одинаковый прогрев. Чётные повторы идут
    Fragment → Compute, нечётные Compute → Fragment. Чётное число повторов
    уравнивает количество первых и вторых запусков.

    GPU timer query измеряет интервал серии проходов; он может включать
    ожидание команд Python. Wall time дополнительно включает отправку команд
    и финальную синхронизацию. Создание ресурсов, компиляция, сброс, прогрев,
    чтение полей, графики и видео находятся **за пределами** обоих интервалов.
    Медиана и диапазон повторов показывают разброс, а не доверительный интервал.

    N×N — число ячеек, текстуры имеют (N+1)×(N+1) узлов.
    Область фиксирована: 0,15×0,15 м. При увеличении N уменьшается dt;
    цена шага не равна цене одинаковой физической длительности. Для неё
    число шагов растёт пропорционально N. Небольшие сетки могут быть ограничены
    отправкой команд из Python. Colab разделяет инфраструктуру: повторите
    эксперимент перед обобщением результата.
    ''')
    add("code", "parameters", '''
    SIZES = [128, 256, 512, 1000, 2000]
    BOUNDARY = "pml"              # либо "pec"
    STEPS = 300
    WARMUP = 300
    REPEATS = 6                   # чётное число, минимум 2
    MAKE_VIDEO = True             # отдельный расчёт, вне бенчмарка
    VIDEO_SIZE = 512
    assert REPEATS >= 2 and REPEATS % 2 == 0
    assert STEPS > 0 and WARMUP > 0 and len(set(SIZES)) == len(SIZES)
    ''')
    add("code", "measurement", '''
    def measure(solver):
        solver.reset()
        for _ in range(WARMUP):
            solver.step()
        ctx.finish()
        query = ctx.query(time=True)
        started = time.perf_counter()
        with query:
            for _ in range(STEPS):
                solver.step()
        ctx.finish()
        wall_seconds = time.perf_counter() - started
        gpu_seconds = query.elapsed / 1e9
        assert gpu_seconds > 0 and wall_seconds > 0
        return dict(gpu_ms_per_step=1000 * gpu_seconds / STEPS,
                    wall_ms_per_step=1000 * wall_seconds / STEPS)

    def compare_fields(fragment, compute):
        errors = []
        # Отдельная проверка после таймера, включая все каналы E и H.
        for left, right in zip(fragment.read_fields(), compute.read_fields()):
            assert left.dtype == right.dtype == np.float32
            assert np.isfinite(left).all() and np.isfinite(right).all()
            for channel in range(left.shape[-1]):
                a, b = left[..., channel], right[..., channel]
                error = float(np.max(np.abs(a - b))) / max(float(np.max(np.abs(a))), 1e-8)
                assert error < 5e-5, error
                errors.append(error)
        return max(errors)

    trials, field_checks = [], []
    for size in SIZES:
        config = TMConfig(size, BOUNDARY)
        solvers = {}
        try:
            solvers["fragment"] = FragmentTM(ctx, config)
            solvers["compute"] = ComputeTM(ctx, config)
            for repeat in range(REPEATS):
                order = ("fragment", "compute") if repeat % 2 == 0 else ("compute", "fragment")
                for position, backend in enumerate(order):
                    timing = measure(solvers[backend])
                    trials.append(dict(size=size, repeat=repeat, position=position,
                                       backend=backend, **timing))
                error = compare_fields(solvers["fragment"], solvers["compute"])
                field_checks.append(dict(size=size, repeat=repeat, max_relative_error=error))
            print(f"{size}×{size}: {REPEATS} пар, поля совпадают в пределах допуска", flush=True)
        finally:
            for solver in solvers.values():
                solver.release()
    ''')
    add("code", "analysis", '''
    summary = []
    for size in SIZES:
        row = dict(size=size)
        for backend in ("fragment", "compute"):
            subset = [t for t in trials if t["size"] == size and t["backend"] == backend]
            for metric in ("gpu", "wall"):
                values = [t[f"{metric}_ms_per_step"] for t in subset]
                row[f"{backend}_{metric}"] = float(np.median(values))
                row[f"{backend}_{metric}_min"] = min(values)
                row[f"{backend}_{metric}_max"] = max(values)
        # >1: Compute быстрее; <1: Fragment быстрее.
        row["speedup_gpu"] = row["fragment_gpu"] / row["compute_gpu"]
        row["speedup_wall"] = row["fragment_wall"] / row["compute_wall"]
        summary.append(row)

    table = "<table><tr><th>N</th><th>Fragment GPU, ms/step</th><th>Compute GPU, ms/step</th><th>F/C GPU</th><th>Fragment wall</th><th>Compute wall</th><th>F/C wall</th></tr>"
    for r in summary:
        table += (f"<tr><td>{r['size']}</td><td>{r['fragment_gpu']:.4f}</td>"
                  f"<td>{r['compute_gpu']:.4f}</td><td>{r['speedup_gpu']:.2f}</td>"
                  f"<td>{r['fragment_wall']:.4f}</td><td>{r['compute_wall']:.4f}</td>"
                  f"<td>{r['speedup_wall']:.2f}</td></tr>")
    display(HTML(table + "</table><p>F/C > 1: Compute быстрее; F/C < 1: Fragment быстрее. Все времена — мс/шаг.</p>"))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for backend, label in (("fragment", "Fragment"), ("compute", "Compute")):
        for ax, metric in zip(axes[:2], ("gpu", "wall")):
            median = np.array([r[f"{backend}_{metric}"] for r in summary])
            low = np.array([r[f"{backend}_{metric}_min"] for r in summary])
            high = np.array([r[f"{backend}_{metric}_max"] for r in summary])
            ax.errorbar(SIZES, median, yerr=[median-low, high-median], marker="o", capsize=3, label=label)
            ax.set(xscale="log", yscale="log", xlabel="N (N × N cells)", ylabel="ms / step",
                   title=f"{metric.upper()}: median and min–max")
            ax.legend()
    axes[2].plot(SIZES, [r["speedup_gpu"] for r in summary], "o-", label="GPU interval")
    axes[2].plot(SIZES, [r["speedup_wall"] for r in summary], "s--", label="Wall")
    axes[2].axhline(1, color="grey", linestyle=":")
    axes[2].set(xscale="log", xlabel="N", ylabel="Fragment time / Compute time", title=">1: Compute faster")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=.25)
    fig.suptitle(f"{ctx.info['GL_RENDERER']}; float32; {BOUNDARY}; {REPEATS} alternating pairs")
    fig.tight_layout()
    fig.savefig(OUTPUT / "comparison.png", dpi=180)
    fig.savefig(OUTPUT / "comparison.svg")
    plt.show()
    ''')
    add("markdown", "video-notes", '''
    ## Визуальное сравнение — отдельный запуск

    Одинаковые сетка, длительность, FPS, разрешение и цветовая шкала.
    Эти видео проверяют поведение поля, а не скорость решателей.
    Рендеринг, чтение RGB и FFmpeg не входят в измерения выше.
    На каждом размере в бенчмарке также сравнивались все численные поля.
    ''')
    add("code", "videos", '''
    videos = {}
    if MAKE_VIDEO:
        config = TMConfig(VIDEO_SIZE, BOUNDARY)
        for backend, render in (("fragment", render_fragment_video), ("compute", render_compute_video)):
            path = OUTPUT / f"{backend}.mp4"
            videos[backend] = render(ctx, config, path, steps=config.default_steps,
                                     stride=max(1, round(20 * VIDEO_SIZE / 1000)),
                                     fps=25, resolution=640, limit=VIDEO_SIZE)
            display(HTML(f"<b>{backend.title()}</b>"))
            display(Video(str(path), embed=True, mimetype="video/mp4", width=640,
                          html_attributes="controls loop playsinline"))
    ''')
    add("code", "artifacts", '''
    report = dict(commit=COMMIT, module_sha256=MODULE_HASHES,
                  opengl={k: ctx.info[k] for k in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")},
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                matplotlib=matplotlib.__version__, moderngl=moderngl.__version__),
                  dtype="float32", boundary=BOUNDARY, sizes=SIZES, steps=STEPS,
                  warmup=WARMUP, repeats=REPEATS, workgroup_size=[16,16,1],
                  tests_run=test_result.testsRun, tests_ok=test_result.wasSuccessful(),
                  trials=trials, field_checks=field_checks, summary=summary, videos=videos,
                  timing_note="GPU interval includes possible Python submission gaps; video excluded")
    (OUTPUT / "comparison.json").write_text(json.dumps(report, indent=2))
    archive_path = OUTPUT / "comparison_results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("comparison.json", "comparison.png", "comparison.svg", "opengl_context.json"):
            archive.write(OUTPUT / name, name)
        for backend in videos:
            for suffix in ("mp4", "npz"):
                path = OUTPUT / f"{backend}.{suffix}"
                archive.write(path, path.name)
    print("COMPARE_NOTEBOOK_OK", archive_path)
    display(FileLink(str(archive_path)))
    ''')
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name="Python 3", language="python", name="python3"),
                                        language_info=dict(name="python"), accelerator="GPU"), nbformat=4, nbformat_minor=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    target = ROOT / "fdtd_compare_colab.ipynb"
    data = json.dumps(build_notebook(), ensure_ascii=False, indent=1) + "\n"
    if args.check:
        assert target.read_text() == data, "Regenerate fdtd_compare_colab.ipynb"
    else:
        target.write_text(data)
