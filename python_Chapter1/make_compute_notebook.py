"""Build a standalone Colab notebook with directly executed compute-shader cells."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import textwrap

ROOT = Path(__file__).resolve().parent


def build_notebook():
    cells = []

    def add(kind, cell_id, source):
        cell = dict(cell_type=kind, id=cell_id, metadata={},
                    source=textwrap.dedent(source).strip().splitlines(keepends=True))
        if cell["source"]:
            cell["source"][-1] += "\n"
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    common = (ROOT / "fragment_fdtd.py").read_text()
    sections = re.split(r"^# %% (\w+)\n", common, flags=re.MULTILINE)
    code = dict(zip(sections[1::2], sections[2::2]))
    source = (ROOT / "compute_fdtd.py").read_text()
    compute_sections = re.split(r"^# %% (\w+)\n", source, flags=re.MULTILINE)
    code.update(zip(compute_sections[1::2], compute_sections[2::2]))
    common_tree = ast.parse(common)
    imports = "\n".join(
        ast.get_source_segment(common, node) for node in common_tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )
    display_shaders = "\n\n".join(
        ast.get_source_segment(common, node) for node in common_tree.body
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in ("VERTEX_SHADER", "COLOR_SHADER")
            for target in node.targets
        )
    )
    checks = (ROOT / "compute_checks.py").read_text()
    test_node = next(n for n in ast.parse(checks).body if isinstance(n, ast.ClassDef))

    add("markdown", "intro", """
    # FDTD на GPU Colab: Compute Shader, float32

    **Выберите GPU-среду Colab (например T4) и выполните все ячейки.**
    Сначала проверяется настоящий NVIDIA OpenGL-контекст. Программный рендерер
    `llvmpipe` не допускается: если аппаратный EGL недоступен, расчёт остановится.

    Это самостоятельный учебный пример Compute Shader и исходная реализация
    для будущего сравнения с Fragment Shader. Здесь реализован
    **TMz FDTD**: компоненты Ez, Hx, Hy, производная гауссова импульса в центре,
    поглощающий PML Беренже либо отражающие PEC-стенки. Рупор TEz пока не входит
    в этот эксперимент. Существующий NumPy-блокнот остаётся отдельным примером.

    Все определения и вызовы видны в ячейках. Python-файлы на VM не создаются.
    NVIDIA GPU считает поля и раскрашивает изображение; CPU передаёт команды,
    формирует значение источника и кодирует видео через FFmpeg.
    """)
    add("code", "packages", """
    # @title 1. Библиотеки OpenGL
    %pip -q install moderngl==5.12.0 glcontext==3.0.0
    """)
    add("code", "imports", imports)
    add("markdown", "egl-notes", """
    ## 2. Проверка аппаратного OpenGL

    EGL создаёт контекст без окна или X-сервера. В проверенной Colab-сессии
    библиотеки NVIDIA находились в `/usr/lib64-nvidia`, но стандартный EGL
    выбирал Mesa/CPU. Ниже явно указывается существующая NVIDIA EGL-библиотека
    через небольшой vendor JSON. Драйвер не переустанавливается.

    Ожидаем `GL_VENDOR = NVIDIA Corporation`, имя GPU в `GL_RENDERER` и
    OpenGL >= 4.3 нужен для compute shaders. Fragment Shader пример
    использует ту же версию API. После неудачного создания Mesa-контекста
    перезапустите ядро перед повторным запуском этого блока.
    """)
    add("code", "egl-function", code["EGL"])
    add("code", "egl-check", """
    # @title Создание NVIDIA EGL-контекста
    WORK_DIR = Path.cwd() / "balanis_compute"
    WORK_DIR.mkdir(exist_ok=True)
    ctx = create_nvidia_context(WORK_DIR)
    print("NVIDIA_EGL_OK")
    """)
    add("markdown", "model-notes", """
    ## 3. Сетка, коэффициенты и небольшой CPU-эталон

    Область всегда имеет размер **0,15×0,15 м**. При увеличении числа ячеек
    уменьшаются `dx` и `dt`, а длительность импульса и число шагов масштабируются
    так, чтобы сохранять физическую временную шкалу. Толщина PML также
    масштабируется вместе с сеткой. Используется исходное возбуждение `voltage/dx`.

    Все поля, текстуры, передаваемые коэффициенты и арифметика шейдеров —
    **float32**. Константы и профили PML предварительно рассчитываются на CPU
    и один раз преобразуются в float32. `TMReference` нужен только для тестов
    на маленькой сетке и не участвует в GPU-замерах.

    Массивы имеют индексы `[i, j]`, texel OpenGL — `(j, i)`.
    Поля упакованы в текстуры: E=(Ez, Ezx, Ezy, 0), H=(Hx, Hy).
    Для удобства они дополнены до одинакового размера (N+1)×(N+1).
    """)
    add("code", "model", code["model"])
    add("markdown", "shader-notes", """
    ## 4. Compute shaders GLSL: один вызов на узел поля

    `layout(local_size_x=16, local_size_y=16)` задаёт 256 потоков в рабочей группе.
    `gl_GlobalInvocationID.xy` даёт координаты texel `(j, i)`. Число групп равно
    `ceil((N+1)/16)` по каждой оси. На сетке 1000×1000 хранятся 1001×1001 узел:
    запускаются 63×63 группы (1008×1008 потоков), а лишние потоки завершаются.

    Как и во Fragment Shader, входные поля читаются через `texelFetch`.
    Выход записывается через `imageStore` в отдельную текстуру. Форматы и
    упаковка E/H, формулы, порядок арифметики и `precise` сохранены одинаковыми.
    Никакой общей памяти рабочих групп и объединения временных шагов здесь нет.

    Один шаг содержит два dispatch:

    1. Старые E/H → новые E и источник.
    2. Новые E и старые H → новые H.

    Между ними `ctx.memory_barrier` делает результаты `imageStore` видимыми
    последующему `texelFetch` и упорядочивает следующие записи. Барьер внутри
    одного GLSL workgroup не синхронизировал бы всю сетку, поэтому этапы разделены
    на два dispatch. Для чтения/сброса текстур на CPU используется отдельный
    `TEXTURE_UPDATE_BARRIER_BIT`. Входные и выходные текстуры чередуются (ping-pong).
    """)
    add("code", "shaders", code["shaders"])
    add("code", "gpu-solver", code["gpu"])
    add("markdown", "test-notes", """
    ## 5. Проверки корректности на GPU

    Проверяем случайные начальные поля и полные траектории относительно NumPy
    float32, обе границы, симметрию, поглощение и сброс состояния. Дополнительный
    тест проверяет сетки, заканчивающиеся внутри рабочей группы 16×16.
    Допуск нормированной максимальной ошибки компоненты — 5×10⁻⁵;
    побитовое совпадение не требуется.
    """)
    add("code", "tests", "import unittest\n" + ast.get_source_segment(checks, test_node))
    add("code", "run-tests", """
    # @title Запуск GPU-тестов
    import sys
    ComputeTests.context = ctx
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ComputeTests)
    )
    if not result.wasSuccessful() or result.testsRun != 5 or result.skipped:
        raise AssertionError("GPU-тесты должны пройти до замеров и создания видео")
    """)
    add("markdown", "parameter-notes", """
    ## 6. Параметры эксперимента

    При сетке 1000×1000 и PML исходные 111 шагов превращаются в 2220 шагов:
    уменьшение клетки требует уменьшения шага времени. Это около 0,555 нс.
    `VIDEO_STEPS=0` выбирает такую исходную физическую длительность.
    FPS влияет только на воспроизведение, `FRAME_STRIDE` — на частоту съёмки.
    """)
    add("code", "parameters", """
    # @title Параметры
    GRID_SIZE = 1000 # @param {type:"integer"}
    BOUNDARY = "pml" # @param ["pml", "pec"]
    BENCHMARK_STEPS = 300 # @param {type:"integer"}
    WARMUP_STEPS = 300 # @param {type:"integer"}
    REPEATS = 3 # @param {type:"integer"}
    VIDEO_STEPS = 0 # @param {type:"integer"}
    FRAME_STRIDE = 20 # @param {type:"integer"}
    FPS = 25 # @param {type:"integer"}
    FIELD_LIMIT = 1000.0 # @param {type:"number"}
    for name in ("GRID_SIZE", "BENCHMARK_STEPS", "WARMUP_STEPS", "REPEATS", "VIDEO_STEPS", "FRAME_STRIDE", "FPS"):
        value = globals()[name]
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} должен быть целым числом")
    if min(BENCHMARK_STEPS, WARMUP_STEPS, REPEATS, FRAME_STRIDE, FPS) < 1 or VIDEO_STEPS < 0:
        raise ValueError("Некорректные параметры")
    if not math.isfinite(FIELD_LIMIT) or FIELD_LIMIT <= 0:
        raise ValueError("FIELD_LIMIT должен быть положительным конечным числом")
    config = TMConfig(GRID_SIZE, BOUNDARY)
    video_steps = VIDEO_STEPS or config.default_steps
    if video_steps < FRAME_STRIDE:
        raise ValueError("Число шагов должно быть не меньше FRAME_STRIDE")
    print(f"{GRID_SIZE}×{GRID_SIZE}, dx={config.dx:.6g} м, dt={config.dt:.6g} с")
    print(f"Рабочие группы: {(GRID_SIZE + 16) // 16}×{(GRID_SIZE + 16) // 16}, по 16×16 потоков")
    print(f"Видео: {video_steps} шагов, {video_steps // FRAME_STRIDE} кадров, "
          f"физическое время {video_steps * config.dt * 1e9:.3f} нс")
    """)
    add("markdown", "timing-notes", """
    ## 7. Замер расчёта без видео

    Инициализация, компиляция шейдеров, прогрев и чтение полей исключены из замера.
    Каждый повтор начинается с нулевых полей. Сохраняются GPU timer query
    и полное синхронизированное время цикла Python→OpenGL.

    **GPU timer query охватывает всю серию проходов** и может включать ожидание
    GPU, пока Python отправляет очередные команды. Поэтому это время данного
    способа запуска решателя, а не изолированная пропускная способность GPU.
    Сравнивать эти числа с отдельным прошлым запуском Fragment Shader нельзя:
    для вывода о скорости нужны оба решателя в одном контексте и одинаковые замеры.
    """)
    add("code", "benchmark-function", code["benchmark"])
    add("code", "run-benchmark", """
    # @title GPU-замеры
    benchmark = benchmark_compute(ctx, config, steps=BENCHMARK_STEPS,
                                   repeats=REPEATS, warmup=WARMUP_STEPS)
    benchmark["opengl"] = {key: ctx.info[key] for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}
    for row in benchmark["repeats"]:
        print(f"Повтор {row['repeat']}: GPU interval {row['gpu_ms_per_step']:.3f} мс/шаг; "
              f"wall {row['wall_ms_per_step']:.3f} мс/шаг; "
              f"{row['million_cell_steps_per_second']:.1f} млн ячеек·шаг/с")
    _ = (WORK_DIR / "benchmark.json").write_text(json.dumps(benchmark, indent=2))
    """)
    add("markdown", "video-notes", """
    ## 8. Видео под ячейкой

    Решатель запускается заново. Отдельный **fragment shader** раскрашивает Ez:
    **синий — отрицательное поле, белый — ноль, красный — положительное**.
    Физические поля не копируются на CPU на каждом шаге: передаются только RGB-кадры
    по `FRAME_STRIDE`. CPU кодирует их в MP4/H.264. Эта работа не входит в замер выше.
    В конце сохраняются финальные поля NPZ.
    """)
    add("code", "color-shaders", display_shaders)
    add("code", "video-function", code["video"])
    add("code", "run-video", """
    # @title Расчёт и показ видео
    from IPython.display import Video, display
    video_path = WORK_DIR / "compute_tm.mp4"
    video_report = render_compute_video(
        ctx, config, video_path, steps=video_steps, stride=FRAME_STRIDE,
        fps=FPS, resolution=800, limit=FIELD_LIMIT,
    )
    print(json.dumps(video_report, indent=2))
    display(Video(str(video_path), embed=True, mimetype="video/mp4", width=720,
                  html_attributes="controls loop playsinline"))
    """)
    add("code", "verify-export", """
    # @title Проверка MP4 и численных полей
    import subprocess
    from fractions import Fraction
    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames",
        "-of", "json", str(video_path),
    ], text=True))["streams"][0]
    assert probe["codec_name"] == "h264"
    assert int(probe["nb_read_frames"]) == video_report["frames"] == video_steps // FRAME_STRIDE
    assert Fraction(probe["r_frame_rate"]) == FPS
    assert (probe["width"], probe["height"]) == (800, 800)
    with np.load(video_path.with_suffix(".npz")) as data:
        for name in ("ez", "ezx", "ezy", "hx", "hy"):
            assert data[name].dtype == np.dtype("float32")
            assert np.isfinite(data[name]).all()
        assert data["ez"].shape == (GRID_SIZE + 1, GRID_SIZE + 1)
        assert int(data["step"]) == video_steps
    report = dict(opengl=benchmark["opengl"], tests_run=result.testsRun,
                  tests_ok=result.wasSuccessful(), video=video_report, ffprobe=probe)
    _ = (WORK_DIR / "verification.json").write_text(json.dumps(report, indent=2))
    print("Видео и float32-поля проверены")
    """)
    add("code", "download", """
    # @title 9. Архив результатов
    DOWNLOAD_RESULTS = False # @param {type:"boolean"}
    import zipfile
    archive_path = WORK_DIR / "compute_results.zip"
    with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("compute_tm.mp4", "compute_tm.npz", "benchmark.json",
                     "verification.json", "opengl_context.json"):
            archive.write(WORK_DIR / name, arcname=name)
    print(f"Архив: {archive_path} ({archive_path.stat().st_size / 1024:.0f} КиБ)")
    if DOWNLOAD_RESULTS:
        from google.colab import files
        files.download(str(archive_path))
    print("COMPUTE_NOTEBOOK_OK")
    """)
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(name="python3", display_name="Python 3", language="python"),
        language_info=dict(name="python"), colab=dict(name="fdtd_compute_colab.ipynb", provenance=[]),
        source_sha256={"fragment_fdtd.py": hashlib.sha256(common.encode()).hexdigest(),
                       "compute_fdtd.py": hashlib.sha256(source.encode()).hexdigest(),
                       "compute_checks.py": hashlib.sha256(checks.encode()).hexdigest()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "fdtd_compute_colab.ipynb"
    content = json.dumps(build_notebook(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not path.exists() or path.read_text() != content:
            raise SystemExit("Run python python_Chapter1/make_compute_notebook.py")
        print("Compute notebook matches its sources")
    else:
        path.write_text(content)
        print(path)


if __name__ == "__main__":
    main()
