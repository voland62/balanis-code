"""Generate the standalone Colab notebook from the maintained solver and tests."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent


def build_notebook():
    cells = []

    def add(kind, cell_id, source, **metadata):
        cell = {"cell_type": kind, "id": cell_id, "metadata": metadata,
                "source": textwrap.dedent(source).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "introduction", """
    # Электромагнитные волны: FDTD на Python · версия 2

    Три примера главы 1 Balanis: прямой перенос MATLAB-кода W. V. Andrew (1995)
    на NumPy. Решаются двумерные уравнения Максвелла на сетке Йи.

    **Запуск в Colab:** выберите обычную **CPU**-среду и **«Выполнить все»**.
    GPU и доступ к Google Drive не нужны. Блокнот самостоятельный: исходники
    решателя и тестов включены ниже, скачивать репозиторий не требуется.
    Все вычисления и тесты выполняются прямо в ячейках текущего Python-ядра.
    Видео появляется сразу под ячейкой расчёта.

    Последовательность: определяем модель → проверяем её тестами → задаём
    параметры → обновляем поля во времени → смотрим распространение волны.

    | Сценарий | Модель | Возбуждение | Шаги / кадры по умолчанию |
    | --- | --- | --- | --- |
    | `tm_open` | Свободное пространство, PML 6 ячеек | Производная гауссиана | 111 / 37 |
    | `tm_box` | Проводящая полость, отражения | Производная гауссиана | 210 / 70 |
    | `te_horn` | Двумерный PEC-рупор, PML 8 ячеек | Синусоида 9,84252 ГГц с плавным включением | 210 / 70 |

    Сетки — 50×50 ячеек для TMz и 100×100 для TEz. Время расчёта задаётся
    `dt = dx/(2c)`; FPS меняет только скорость воспроизведения видео.
    У рупора возбуждение непрерывное, а не одиночный импульс.
    """)
    add("code", "dependencies", '''
    # @title 1. Подготовка окружения
    import importlib.util
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import sys

    missing = [name for name in ("numpy", "matplotlib")
               if importlib.util.find_spec(name) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", *missing], check=True)
    if not all(shutil.which(name) for name in ("ffmpeg", "ffprobe")):
        if importlib.util.find_spec("google.colab") is None:
            raise RuntimeError("Установите ffmpeg и ffprobe перед запуском блокнота.")
        subprocess.run(["apt-get", "update", "-qq"], check=True)
        subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=True)

    WORK_DIR = Path.cwd() / "balanis_fdtd"
    WORK_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR = WORK_DIR / "output"
    OUTPUT_DIR.mkdir(exist_ok=True)
    import numpy as np
    import matplotlib
    print("Python:", sys.version.split()[0])
    print("NumPy:", np.__version__, "Matplotlib:", matplotlib.__version__)
    print(subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0])
    print("Рабочая папка:", WORK_DIR)
    ''')
    hashes = {}
    sources = {}
    for name, class_name in (("fdtd.py", "Simulation"), ("test_fdtd.py", "FDTDTests")):
        source = (ROOT / name).read_text()
        hashes[name] = hashlib.sha256(source.encode()).hexdigest()
        node = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.ClassDef) and node.name == class_name)
        sources[class_name] = ast.get_source_segment(source, node)

    add("markdown", "model-notes", """
    ## 2. Решатель уравнений Максвелла

    Электрические и магнитные компоненты расположены на смещённой сетке Йи.
    На каждом временном шаге пространственные разности одного поля обновляют
    другое. Шаг времени `dt = dx/(2c)` удовлетворяет условию устойчивости
    для используемой квадратной двумерной сетки.

    Ниже определён класс `Simulation`. Создание `Simulation("tm_open")`
    задаёт начальные поля и поглощающую границу; `sim.step()` выполняет один
    шаг расчёта. `sim.field` возвращает отображаемое поле.
    Методы `_pml` и `_horn` задают поглощающие слои и проводящие стенки рупора.

    Формулы, индексы и порядок обновлений сохранены из MATLAB-оригинала.
    Для рупора отображается `log10(abs(Ey + 1e-6))`, для TMz — знаковое `Ez`.
    """)
    add("code", "solver-source", '# @title Определение решателя\n' + sources["Simulation"])
    add("markdown", "plot-notes", """
    ## 3. Отображение поля

    Две небольшие функции ниже создают цветовую карту и обновляют её для
    очередного кадра. Сам расчёт полей находится отдельно — в `sim.step()`.
    Строки массива направлены по вертикали, столбцы — по горизонтали,
    как в исходном MATLAB `pcolor`. Цветовая шкала фиксирована во времени.
    """)
    add("code", "plot-functions", '''
    # @title Функции визуализации
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from IPython.display import Markdown, Video, display

    def create_field_plot(sim):
        fig, ax = plt.subplots(figsize=(8, 7), layout="constrained")
        limits = (-6, 3) if sim.te else (-50, 50)
        rows, cols = sim.field.shape
        mesh = ax.pcolormesh(
            np.arange(1, cols + 1), np.arange(1, rows + 1), sim.display_field(),
            shading="gouraud", cmap="jet", vmin=limits[0], vmax=limits[1],
        )
        ax.set(xlim=(0, sim.size), ylim=(0, sim.size), aspect="equal",
               xlabel="j (cell index, MATLAB convention)",
               ylabel="i (cell index, MATLAB convention)")
        fig.colorbar(mesh, ax=ax, label="log10(|Ey + 10⁻⁶|)" if sim.te else "Ez (V/m)")
        title = ax.set_title("")
        return fig, mesh, title

    def update_field_plot(sim, mesh, title):
        mesh.set_array(sim.display_field())
        title.set_text(
            f"{sim.scenario.upper()} · step {sim.n} · t = {sim.n * sim.dt * 1e9:.3f} ns"
        )
    ''')
    add("markdown", "test-notes", """
    ## 4. Численные тесты

    Проверяем знаки и индексы уравнений независимыми циклами по ячейкам,
    геометрию стенок, симметрию импульса и поглощение PML.
    Тесты вызывают тот же класс `Simulation`, который определён выше.
    """)
    add("code", "test-source", 'import unittest\n\n' + sources["FDTDTests"])
    add("code", "numerical-tests", '''
    # @title Запуск тестов прямо в ядре блокнота
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(FDTDTests)
    test_result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    if not test_result.wasSuccessful():
        raise AssertionError("Численные тесты не прошли — проверьте решатель выше")
    ''')
    add("code", "parameters", '''
    # @title 5. Параметры моделирования
    SCENARIO = "all" # @param ["all", "tm_open", "tm_box", "te_horn"]
    STEPS = 0 # @param {type:"integer"}
    FRAME_STRIDE = 3 # @param {type:"integer"}
    FPS = 7 # @param {type:"integer"}

    # STEPS = 0: исходная длительность каждого сценария.
    # При изменении параметров повторите выполнение этой и следующих ячеек.
    DEFAULT_STEPS = {"tm_open": 111, "tm_box": 210, "te_horn": 210}
    if SCENARIO not in ("all", *DEFAULT_STEPS):
        raise ValueError("Неизвестный сценарий")
    for name, value in (("STEPS", STEPS), ("FRAME_STRIDE", FRAME_STRIDE), ("FPS", FPS)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"{name} должен быть целым числом")
    if STEPS < 0 or FRAME_STRIDE < 1 or FPS < 1:
        raise ValueError("STEPS должен быть >= 0, FRAME_STRIDE и FPS — >= 1")
    scenarios = list(DEFAULT_STEPS) if SCENARIO == "all" else [SCENARIO]
    if any((STEPS or DEFAULT_STEPS[name]) < FRAME_STRIDE for name in scenarios):
        raise ValueError("Число шагов должно быть не меньше интервала между кадрами")
    print("Будут рассчитаны:", ", ".join(scenarios))
    ''')
    add("markdown", "simulation-notes", """
    ## 6. Расчёт и видео

    В следующей ячейке создаётся `sim = Simulation(name)`. Временной цикл
    вызывает `sim.step()` и снимает кадр каждые `FRAME_STRIDE` шагов.
    `FFMpegWriter` кодирует кадры в MP4, а `Video(..., embed=True)` вставляет
    видео в вывод этой же ячейки.

    MP4, финальный снимок PNG и поля NPZ сохраняются для скачивания.
    Можно изменить параметры выше и повторить расчёт: каждый сценарий
    начинается с новых нулевых полей. Вывод ячейки заменится новым видео.
    """)
    add("code", "simulate", '''
    # @title Распространение волны и видео
    import time
    generated = []
    for name in scenarios:
        start = time.perf_counter()
        sim = Simulation(name)
        steps = STEPS or sim.default_steps
        video_path = OUTPUT_DIR / f"{name}.mp4"
        fig, mesh, title = create_field_plot(sim)
        writer = FFMpegWriter(fps=FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
        try:
            with writer.saving(fig, str(video_path), dpi=100):
                for step in range(steps):
                    sim.step()  # Один шаг решения уравнений Максвелла.
                    if sim.n % FRAME_STRIDE == 0:
                        update_field_plot(sim, mesh, title)
                        writer.grab_frame()
            update_field_plot(sim, mesh, title)
            fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=150)
            sim.save(OUTPUT_DIR / f"{name}.npz")
        finally:
            plt.close(fig)
        generated.append(name)
        print(f"{name}: {steps} шагов, {steps // FRAME_STRIDE} кадров, "
              f"{time.perf_counter() - start:.1f} с", flush=True)
        display(Markdown(f"### {name}"))
        display(Video(str(video_path), embed=True, mimetype="video/mp4", width=720,
                      html_attributes="controls loop playsinline"))
    ''')
    add("code", "verify-videos", '''
    # @title 7. Проверка видео и сохранённых полей
    import json
    from fractions import Fraction
    report = []
    for name in generated:
        video = OUTPUT_DIR / f"{name}.mp4"
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames",
            "-of", "json", str(video),
        ], text=True))["streams"][0]
        expected_steps = STEPS or DEFAULT_STEPS[name]
        assert probe["codec_name"] == "h264", probe
        assert int(probe["nb_read_frames"]) == expected_steps // FRAME_STRIDE, probe
        assert Fraction(probe["r_frame_rate"]) == FPS, probe
        assert (probe["width"], probe["height"]) == (800, 700), probe
        with np.load(OUTPUT_DIR / f"{name}.npz") as fields:
            assert int(fields["step"]) == expected_steps
            assert str(fields["scenario"]) == name
            for key in (("ex", "ey", "hz") if name == "te_horn" else ("ez", "hx", "hy")):
                assert np.isfinite(fields[key]).all(), (name, key)
            field = fields["ey" if name == "te_horn" else "ez"]
            maximum = float(np.max(np.abs(field)))
        report.append(dict(scenario=name, steps=expected_steps,
                           frames=int(probe["nb_read_frames"]), fps=FPS,
                           maximum_field=maximum, video_bytes=video.stat().st_size))
        print(f"OK: {name}, {probe['nb_read_frames']} кадров, {FPS} fps, max |field|={maximum:.8g}")
    _ = (OUTPUT_DIR / "verification.json").write_text(json.dumps(report, indent=2))
    ''')
    add("markdown", "download-notes", """
    Видео встроены в вывод ячеек. Финальные PNG и численные массивы NPZ находятся
    в `balanis_fdtd/output`; их также можно скачать через панель «Файлы» Colab.
    Финальный снимок `tm_open` почти однородный: к концу расчёта волна уже уходит
    в поглощающую границу — процесс виден на видео.

    Следующая ячейка создаёт ZIP текущего запуска. Чтобы скачать его на компьютер,
    включите `DOWNLOAD_RESULTS` и выполните эту ячейку. Файлы на VM временные:
    сохраните нужные результаты перед завершением сессии.
    """)
    add("code", "download", '''
    # @title 8. Архив результатов
    DOWNLOAD_RESULTS = False # @param {type:"boolean"}
    import zipfile
    archive = WORK_DIR / "fdtd_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for name in generated:
            for suffix in ("mp4", "png", "npz"):
                path = OUTPUT_DIR / f"{name}.{suffix}"
                z.write(path, arcname=path.name)
        z.write(OUTPUT_DIR / "verification.json", arcname="verification.json")
    print(f"Архив: {archive} ({archive.stat().st_size / 1024:.0f} КиБ)")
    if DOWNLOAD_RESULTS:
        from google.colab import files
        files.download(str(archive))
    print("FDTD_NOTEBOOK_OK")
    ''')
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"colab": {"name": "fdtd_colab.ipynb", "provenance": [],
                                    "collapsed_sections": []},
                         "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"},
                         "source_sha256": hashes, "fdtd_notebook_version": 2},
            "cells": cells}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if notebook is out of date")
    args = parser.parse_args()
    target = ROOT / "fdtd_colab.ipynb"
    content = json.dumps(build_notebook(), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not target.exists() or target.read_text() != content:
            raise SystemExit("Notebook is out of date; run python python_Chapter1/make_notebook.py")
        print("Notebook matches solver and tests")
    else:
        target.write_text(content)
        print(target)


if __name__ == "__main__":
    main()
