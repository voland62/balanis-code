# FDTD: примеры главы 1 на Python

Прямой перенос расчётов W. V. Andrew (1995) из `../matlab_Chapter1` на NumPy.
Решаются двумерные уравнения Максвелла методом FDTD на сетке Йи.
Специализированный электромагнитный пакет не требуется.

Из корня репозитория войдите в окружение `nix develop` (либо используйте
`direnv reload`). Нужны Python, NumPy, Matplotlib и FFmpeg; они указаны в `flake.nix`.

```bash
python python_Chapter1/fdtd.py tm_open --output python_Chapter1/output/tm_open.mp4
python python_Chapter1/fdtd.py tm_box --output python_Chapter1/output/tm_box.mp4
python python_Chapter1/fdtd.py te_horn --output python_Chapter1/output/te_horn.mp4
```

Сценарии и исходные параметры:

| Сценарий | Поля | Область | Возбуждение | Шаги / кадры |
| --- | --- | --- | --- | --- |
| `tm_open` | Ez, Hx, Hy | 50×50 ячеек, PML 6 ячеек | Производная гауссиана в центре | 111 / 37 |
| `tm_box` | Ez, Hx, Hy | 50×50 ячеек, проводящие стенки | Производная гауссиана в центре | 210 / 70 |
| `te_horn` | Ex, Ey, Hz | 100×100 ячеек, PEC-рупор, PML 8 ячеек | Плавно включаемая синусоида 9,84252 ГГц | 210 / 70 |

В TM-сценариях dx = 0,003 м, в рупоре dx = λ/12; во всех случаях dt = dx/(2c).
Для квадратной двумерной сетки это удовлетворяет условию устойчивости Куранта.
Поля хранятся в float64 на смещённых сетках: например, Ez имеет размер 51×51,
Hx — 51×50, Hy — 50×51. Источник жёсткий: присваивает значение поля,
а не прибавляет его. Сохранены порядок обновлений, коэффициенты и раздельные
компоненты PML Беренже, включая особенности размещения слоёв оригинала.

Дополнительные параметры:

```bash
# Более длинное видео, 25 кадров/с, кадр через каждые два шага расчёта:
python python_Chapter1/fdtd.py tm_open --steps 600 --stride 2 --fps 25

# Только расчёт; сохранить финальные поля и изображение:
python python_Chapter1/fdtd.py te_horn --no-video \
  --data python_Chapter1/output/horn.npz --snapshot python_Chapter1/output/horn.png

python python_Chapter1/fdtd.py --help
```

`--fps` меняет скорость воспроизведения, но не физический шаг времени.
По умолчанию кадр снимается после каждого третьего шага, FPS = 7. Число кадров —
целая часть `steps / stride`; остаточные шаги входят в финальные NPZ и PNG.
Экспорт MP4/H.264 выполняется последовательно через FFmpeg, без накопления кадров.
Рендеринг использует Agg и не требует графического рабочего стола. Видео не
открывается автоматически и не дублируется ради повторов воспроизведения.

Отображается Ez со шкалой [-50, 50] либо `log10(abs(Ey + 1e-6))` со шкалой [-6, 3].
Формула рупора сохранена буквально: добавление epsilon находится внутри модуля.
Используется интерполяция цветов Gouraud и палитра jet. Как в MATLAB `pcolor`,
столбцы массива идут горизонтально, строки — вертикально; координаты рисунка
следуют исходной нумерации от 1 с пределами осей [0, size]. Добавлены цветовая
шкала и физическое время. Совпадение пикселей со старым MATLAB не предполагается.

NPZ содержит финальные массивы полей и `step`, `dx`, `dt`, `scenario`, `npml`.
Для загрузки: `numpy.load("python_Chapter1/output/horn.npz")`.
Старые MATLAB-файлы `.mat` содержали захваченные кадры, а не численные поля;
для сравнения решателей нужен отдельный экспорт полей из MATLAB/Octave.

Проверки:

```bash
python -m unittest discover -s python_Chapter1 -v
```

Проверяются обновления TMz/TEz независимыми скалярными циклами, индексы и
коэффициенты PML, точные позиции проводящих стенок рупора, симметрия импульса,
нулевое электрическое поле на PEC и поглощение уходящей волны.
Сравнение с выполнением MATLAB/Octave не проводилось.

## Google Colab — версия 2

Решатель `Simulation`, функции визуализации и класс тестов определены
непосредственно в ячейках. Тесты запускаются через `unittest.TextTestRunner`,
а расчёт явно вызывает `sim.step()` во временном цикле. MP4-плееры появляются
под ячейкой расчёта. На диске сохраняются только результаты; Python-файлы
для запуска из блокнота не создаются.

Первая версия с запуском Python-процессов сохранена тегом `colab-v1`
(коммит `e3beb43`).

Откройте `python_Chapter1/fdtd_colab.ipynb` в Colab через вкладку GitHub,
выберите CPU-среду и выполните все ячейки. По умолчанию рассчитываются все три
сценария, проходят численные тесты и создаются встроенные MP4-плееры.
Параметры сценария, длительности и FPS доступны в отдельной ячейке.
Финальные PNG, поля NPZ и MP4 можно скачать одним ZIP-архивом.
Блокнот самостоятельный: он содержит решатель и тесты и не требует
клонирования репозитория, Google Drive или авторизации CLI в браузере Colab.

После изменений `fdtd.py` или `test_fdtd.py` обновите блокнот:

```bash
python python_Chapter1/make_notebook.py
python python_Chapter1/make_notebook.py --check
```

Генератор переносит определения классов из исходников непосредственно
в ячейки, без обёрток CLI и импортов локальных модулей. Он сохраняет чистый
блокнот без результатов выполнения. Чтобы проверить
его через установленный `colab-cli`, из Nix-окружения выполните:

```bash
colab new -s balanis-fdtd-check
colab exec -s balanis-fdtd-check -f python_Chapter1/fdtd_colab.ipynb --timeout 600
colab download -s balanis-fdtd-check /content/balanis_fdtd/fdtd_results.zip ./fdtd_results.zip
colab stop -s balanis-fdtd-check
```

CLI создаёт рядом `fdtd_colab_output.ipynb` с результатами; он не нужен для
импорта чистого блокнота из GitHub. При проверке лучше выполнять копию в
игнорируемой папке `output/`, чтобы результаты не попадали в Git.

Проверка результатов CLI после скачивания ZIP (включая ошибки ячеек,
встроенные видео и сравнение всех численных полей с локальным расчётом):

```bash
python python_Chapter1/verify_colab.py \
  python_Chapter1/output/colab_v2_check/fdtd_colab_output.ipynb \
  python_Chapter1/output/colab_v2_check/fdtd_results.zip
```

Одного кода завершения CLI недостаточно: в проверенной версии он может вернуть
0 даже при ошибке внутри ячейки. `verify_colab.py` проверяет сохранённые выводы
и завершает работу с ошибкой, если блокнот выполнился не полностью.

Версия 1 проверена 29 октября 2026 года на CPU-сессии Colab: Python 3.13.18,
NumPy 2.1.3, Matplotlib 3.10.0, FFmpeg 6.1.1. При исходных параметрах
все четыре теста прошли, в ячейках нет ошибок, встроенные MP4 совпали со
скачанными файлами. Все сохранённые компоненты полей трёх сценариев
совпали с локальным расчётом без численных расхождений.

Версия 2 повторно проверена через Colab CLI на CPU с теми же версиями
библиотек: четыре теста прошли прямо в ядре блокнота, все девять кодовых
ячеек выполнены без ошибок. Все три встроенных видео находятся в выводе
ячейки расчёта; число кадров — 37, 70 и 70. Все сохранённые поля совпали
с локальным решателем без численных расхождений.
Выполненная копия и ZIP находятся в `output/colab_v2_check/` и исключены из Git.

## Fragment Shader на GPU Colab

Отдельный эксперимент: `fdtd_fragment_colab.ipynb`. Выберите GPU (например T4)
и выполните все ячейки. Устанавливаются только ModernGL 5.12.0 и glcontext 3.0.0;
NumPy, FFmpeg и ffprobe используются из Colab.

Сначала блокнот создаёт NVIDIA EGL/OpenGL 4.3 контекст. В проверенной среде
NVIDIA-библиотеки есть в `/usr/lib64-nvidia`, но EGL по умолчанию выбирает
Mesa llvmpipe (CPU). Блокнот указывает существующую NVIDIA-библиотеку через
vendor JSON в своей рабочей папке; системные библиотеки и драйвер не меняются.
Если аппаратный NVIDIA-контекст недоступен, проверка завершается ошибкой.
После предыдущих опытов с Mesa может понадобиться перезапуск ядра.

Реализован TMz: Ez/Hx/Hy, производная гауссиана, PML либо PEC.
TEz-рупор в этот этап не входит. Поля и коэффициенты в текстурах float32;
два фрагментных прохода на временной шаг используют чередующиеся текстуры.
Существующие `fdtd.py` и `fdtd_colab.ipynb` не изменяются этим экспериментом.

Область остаётся 0,15×0,15 м. При увеличении сетки уменьшается dt,
масштабируются длительность импульса, число шагов и толщина PML.
По умолчанию: 1000×1000, PML, 2220 шагов, 111 кадров, 25 FPS.
Цветовая карта: отрицательные значения синие, ноль белый, положительные красные.
При сетке 1000×1000 шкала по умолчанию ±1000 В/м; исходник использует
возбуждение voltage/dx, поэтому при уменьшении dx амплитуда источника растёт.

Четыре теста выполняются на GPU: случайные поля и полные траектории
сравниваются с NumPy float32; проверяются симметрия, поглощение и сброс состояния.
CPU-эталон также проверен относительно исходного NumPy-решателя на сетке 50×50.
Предварительное вычисление коэффициентов выполняется на CPU, затем они
преобразуются в float32; это не входит в измерения GPU-решателя.

Замеры расчёта исключают создание ресурсов, компиляцию, прогрев и чтение
полей на CPU. Печатаются все повторы GPU timer query и полное время цикла.
GPU timer query охватывает серию проходов и может включать промежутки ожидания,
пока Python отправляет команды. Это не изолированный замер инструкций шейдера.
Видео создаётся отдельным запуском: GPU раскрашивает поле, выбранные RGB-кадры
копируются на CPU и кодируются FFmpeg. Эти расходы не входят в замер решателя.

Исходники для сопровождения расположены в `fragment_fdtd.py` и
`fragment_checks.py`. Генератор вставляет определения непосредственно в ячейки;
Python-файлы на VM не создаются.

```bash
python python_Chapter1/make_fragment_notebook.py
python python_Chapter1/make_fragment_notebook.py --check
```

Проверка через CLI (лучше выполнять копию блокнота в `output/fragment_check/`):

```bash
mkdir -p python_Chapter1/output/fragment_check
cp python_Chapter1/fdtd_fragment_colab.ipynb python_Chapter1/output/fragment_check/
colab new -s balanis-fragment-check --gpu T4
colab exec -s balanis-fragment-check -f python_Chapter1/output/fragment_check/fdtd_fragment_colab.ipynb --timeout 600
colab download -s balanis-fragment-check /content/balanis_fragment/fragment_results.zip python_Chapter1/output/fragment_check/fragment_results.zip
colab stop -s balanis-fragment-check
python python_Chapter1/verify_fragment_colab.py \
  python_Chapter1/output/fragment_check/fdtd_fragment_colab_output.ipynb \
  python_Chapter1/output/fragment_check/fragment_results.zip
```

Проверяющий скрипт обнаруживает ошибки ячеек независимо от кода возврата CLI,
проверяет NVIDIA renderer, GPU-тесты, встроенное видео, отчёт и float32-поля.

Проверено через Colab CLI 29 октября 2026 года после перезапуска ядра:
Tesla T4, NVIDIA 580.82.07, OpenGL 4.3.0, все 16 кодовых ячеек без ошибок,
четыре GPU-теста прошли. При сетке 1000×1000, PML, прогреве 300 шагов
и трёх повторах по 300 шагов получено 0,2131 / 0,3498 / 0,1967 мс/шаг
(медиана GPU interval 0,2131 мс/шаг; wall 0,2155 мс/шаг).
Отдельное создание видео: 2220 шагов, 111 кадров, 25 FPS, около 4,2 с.
Это результаты конкретного запуска, без сравнения с compute shader.
Архив и выполненный блокнот сохранены в игнорируемой папке `output/fragment_check/`.

## Учебный Compute Shader

`fdtd_compute_colab.ipynb` — самостоятельный учебный блокнот. Выберите GPU
в Colab и выполните все ячейки. Определения решателя, GLSL, тесты, замеры
и видео находятся прямо в ячейках. Файлы `.py` и репозиторий для запуска
блокнота скачивать не нужно. Существующие Fragment и NumPy блокноты не меняются.

Базовая модель и условия совпадают с Fragment Shader: TMz, float32,
область 0,15×0,15 м, PML/PEC, те же коэффициенты и текстуры RGBA32F/RG32F.
На шаг приходятся два compute dispatch: E, затем H. Чтение через `texelFetch`,
запись через `imageStore`. Рабочая группа 16×16; для 1001×1001 узла поля
запускаются 63×63 группы с проверкой выхода потоков за пределы текстуры.
Общая память рабочих групп и объединение временных шагов не используются.

Между dispatch вызываются барьеры `TEXTURE_FETCH_BARRIER_BIT` и
`SHADER_IMAGE_ACCESS_BARRIER_BIT`; перед чтением/сбросом текстур на CPU —
`TEXTURE_UPDATE_BARRIER_BIT`. Внутригрупповой `barrier()` не заменил бы
синхронизацию между этапами на всей сетке.

Пять GPU-тестов проверяют случайные поля, полные траектории относительно
NumPy float32, PML/PEC, симметрию, поглощение, сброс и частичные рабочие группы.
Формат замеров и визуализация совпадают с Fragment Shader. Самостоятельные
замеры из разных Colab-сессий не являются контролируемым сравнением скорости.

Исходники сопровождения: `compute_fdtd.py`, `compute_checks.py` и
`make_compute_notebook.py`. Модуль `compute_fdtd.py` импортирует общую модель,
EGL и шейдер раскраски из `fragment_fdtd.py`; оба модуля нужны для обычного
Python-импорта будущим сравнителем. Генератор встраивает общие определения
в учебный блокнот, поэтому сам `.ipynb` остаётся самостоятельным.

```bash
python python_Chapter1/make_compute_notebook.py
python python_Chapter1/make_compute_notebook.py --check
```

Проверка через Colab CLI:

```bash
mkdir -p python_Chapter1/output/compute_check
cp python_Chapter1/fdtd_compute_colab.ipynb python_Chapter1/output/compute_check/
colab new -s balanis-compute-check --gpu T4
colab exec -s balanis-compute-check -f python_Chapter1/output/compute_check/fdtd_compute_colab.ipynb --timeout 600
colab download -s balanis-compute-check /content/balanis_compute/compute_results.zip python_Chapter1/output/compute_check/compute_results.zip
colab stop -s balanis-compute-check
python python_Chapter1/verify_compute_colab.py \
  python_Chapter1/output/compute_check/fdtd_compute_colab_output.ipynb \
  python_Chapter1/output/compute_check/compute_results.zip
```

Compute-блокнот проверен через Colab CLI 29 октября 2026 года после перезапуска
ядра: Tesla T4, NVIDIA 580.82.07, OpenGL 4.3.0. Все 17 кодовых ячеек выполнены
без ошибок, пять GPU-тестов прошли. На сетке 1000×1000 получено 2220 шагов,
111 кадров при 25 FPS; все финальные поля совпали с сохранённым Fragment
расчётом без численных расхождений. В данном запуске медиана GPU interval
составила 0,3367 мс/шаг, wall — 0,3430 мс/шаг; создание видео — около 4,1 с.
Это запись отдельного запуска, не вывод о сравнительной скорости решателей.
Артефакты находятся в игнорируемой папке `output/compute_check/`.
