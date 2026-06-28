# Compressive Sensing for Hyperspectral Images

This project is a research toolkit for compressing and reconstructing
hyperspectral images (HSIs). Its main focus is compressive sensing: representing
HSI signals in a sparse basis, taking fewer linear measurements than the
original signal length, and recovering the signal through sparse regression.

The main compressive-sensing compressors are:

- **HCS1D**: applies compressive sensing along one HSI axis, typically the
  spectral axis.
- **HCS3D**: applies separable measurement and sparse-basis transforms across
  all three HSI dimensions.
- **Hybrid**: combines spectral compressive sensing with spatial predictive
  coding inspired by CCSDS-123.

Analytic bases such as the DCT can be used directly. Sparse dictionaries can
also be learned from representative HSI spectra using K-SVD. CCSDS-123 is
included as a conventional reference compressor.

## Setup

Install the dependencies and run commands from the project root:

```bash
pip install -r requirements.txt
```

## HSI Preprocessing

AVIRIS scenes can be loaded, filtered, trimmed, divided into sections, and
saved in the project's HSI format:

```python
from src import io, preprocessing

hsi = io.load_aviris_folder("data/raw/JasperRidge")
hsi = preprocessing.filter_spectral_bands(
    hsi,
    remove_ranges=[(104, 108), (150, 163)],
    remove_bands=[220],
)
hsi = preprocessing.trim_borders(hsi, black_value=-50)

for section in preprocessing.crop_hsi_sections(hsi, (256, 256)):
    name = (
        f"JasperRidge_r{section.metadata.section_row}"
        f"_c{section.metadata.section_col}"
    )
    io.save_hsi(section, "data/sections/JasperRidge", name)
```

The configurable version of this workflow is
`scripts/preprocess_aviris.py`.

## Dictionary Learning

Sample spectral training signals and train a K-SVD dictionary:

```python
from src import io, preprocessing, dictionary_trainers
from src.core.dictionary import Axis
from src.pipeline.runner import Runner

hsi = io.load_hsi("data/sections/JasperRidge", "JasperRidge_r1_c1")
signals = preprocessing.sample_training_signals(
    hsi,
    num_signals=5000,
    axis=Axis.SPECTRAL,
    seed=42,
)

config = dictionary_trainers.K_SVDConfig(K=400, T_0=5)
trainer = dictionary_trainers.K_SVD(config)
result = Runner().run_dictionary_training(
    signals,
    trainer,
    experiment="spectral_dictionary",
)

io.save_dictionary(
    result.dictionary,
    "resources/dictionaries",
    "jasper_spectral_400",
)
```

For dataset splits and larger experiments, see
`scripts/create_training_signals.py` and `scripts/train_dictionary.py`.

## Compression

Configure a compressor and run the complete compression, reconstruction, and
metric pipeline:

```python
from src import io, compressors
from src.core.dictionary import Axis
from src.pipeline.runner import Runner
from src.ui.console.callback import ConsoleCallback

hsi = io.load_hsi("data/sections/JasperRidge", "JasperRidge_r11_c1")
learned_basis = (
    "LEARNED:directory=resources/dictionaries,"
    "name=jasper_spectral_400.npz"
)

config = compressors.HCS1DConfig(
    K=5,
    sr=0.1,
    axis=Axis.SPECTRAL,
    Phi="BERNOULLI",
    Psi=learned_basis,
)
compressor = compressors.HCS1D(config)

result = Runner(callbacks=[ConsoleCallback()]).run_compression(
    hsi,
    compressor,
    experiment="hcs1d_example",
)
```

`result` contains the compressed representation, reconstructed HSI, run
metadata, and metrics such as RMSE, PSNR, spectral angle, compression ratio,
and runtime. CSV and artifact logger callbacks can be added to `Runner` to save
experiment results.

Editable single-run and parameter-sweep examples are available in
`scripts/run_compression.py` and `scripts/run_compression_sweep.py`.

## GUI

Launch the desktop GUI with:

```bash
python -m src.ui.gui.app
```

The GUI provides a shared workspace for original, compressed, and reconstructed
HSIs. From it you can:

1. Load HSI or compressed files into the workspace.
2. Select one or more HSIs and configure a compressor.
3. Run a single configuration or a parameter sweep.
4. Review compression metrics while other runs continue.
5. Visualize RGB images, bands, spectra, and histograms.
6. Load and filter experiment logs in the data-analysis tab.
7. Pop plots into separate windows for closer inspection.

Run artifacts and logs can be saved under `results/` and loaded back into the
workspace for comparison.
