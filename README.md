# Compressive Sensing for Hyperspectral Images

This is a research project for testing hyperspectral image (HSI) compression
using compressive sensing. The project provides a convinent framework for testing
different compressors, some compression algorithms, a dictionary learning algorithm and
data analysis functions.

---

## Core Data Objects

### `HSI`

An hyperspectral image is a 3D image consiting of 2 spatial axes and 1 spectral axis. 
In our framework, such images are represented as HSI objects:

```python
HSI(
    data: np.ndarray,
    metadata: HSIMetadata
)
```

The data cube is a 3D numpy array. We use the following convention:

```text
(height, width, bands)
```

---

### `HSIMetadata`

Stores metadata required to describe and reconstruct an HSI.

Includes:

* shape
* wavelengths
* bit depth
* sensor
* scene ID
* scene name
* section index
* additional attributes

---

### `CompressedHSI`

Represents a compressed hyperspectral image.

```python
CompressedHSI(
    bitstream: bytes,
    metadata: HSIMetadata,
    side_information: dict
)
```

`side_information` contains compressor-specific reconstruction information, such as seeds, quantization parameters, measurement shapes, or BER protection masks.

---

### `Dictionary`

Represents a learned dictionary matrix.

```python
Dictionary(
    data: np.ndarray,
    axis: Axis,
    name: str | None = None
)
```

Dictionary shape convention:

```text
(signal_length, num_atoms)
```

---

### `TrainingSignals`

Represents signals extracted from one or more HSIs for dictionary training.

```python
TrainingSignals(
    data: np.ndarray,
    axis: Axis,
    sources: list[HSIMetadata],
    metadata: dict
)
```

Signal matrix convention:

```text
(signal_length, num_signals)
```

---

## Interfaces

### Compressor Interface

All compressors inherit from `Compressor`.

```python
class Compressor:
    name: str
    Config: CompressorConfig

    def compress(self, hsi: HSI) -> CompressedHSI:
        ...

    def decompress(self, compressed: CompressedHSI) -> HSI:
        ...
```

Implemented compressors:

* `HCS1D`
* `HCS3D`
* `CCSDS123`

---

### Dictionary Trainer Interface

All dictionary trainers inherit from `DictionaryTrainer`.

```python
class DictionaryTrainer:
    name: str
    Config: DictionaryTrainerConfig

    def fit(self, signals: TrainingSignals):
        ...
```

Current dictionary trainer:

* `K_SVD`

The current implementation returns:

```python
dictionary, coefficients
```

---

### Metric Interface

Metrics are plugin-style objects.

```python
class Metric:
    name: str
    short_name: str
    unit: str | None

    def compute(self, target) -> MetricResult:
        ...
```

Metric registration uses `short_name`.

---

## Compression Metrics

Implemented compression metrics:

* RMSE
* PSNR
* SAM
* Compression Rate
* Compression Time
* Decompression Time

Timing metrics are added by the runner.

---

## Dictionary Learning Metrics

Implemented dictionary metrics:

* Representation Error
* Mean Sparsity
* Dictionary Coherence
* Training Time

Timing metrics are added by the runner.

---

## Runner

The runner orchestrates experiments.

It is responsible for:

* calling compressors and dictionary trainers
* measuring execution time
* computing metrics
* creating result objects
* emitting status updates

It does not:

* print output
* save files
* implement algorithms
* know about GUI or console UI

---

## Run Results

### `CompressionRunResult`

Stores:

* original HSI
* compressed HSI
* reconstructed HSI
* run metadata
* metrics

---

### `DictionaryTrainingResult`

Stores:

* training signals
* sparse coefficients
* trained dictionary
* run metadata
* metrics

---

### `RunMetadata`

Stores metadata about an experiment run:

* timestamp
* machine name
* user-defined tags

---

## Runner/UI Boundary

The UI controls the runner.

The runner reports status updates through a callback.

```text
UI → Runner → Algorithms / Metrics
Runner → status callback → UI
```

The runner emits `RunStatus` events, and the console UI handles them.

This allows the same runner to later support:

* console UI
* CLI
* GUI
* silent mode
* logging hooks

---

## Console UI

The console UI currently provides:

* status/progress display using `tqdm`
* compression run headers
* compression result summaries
* dictionary learning summaries

Formatting is kept outside the runner.

---

## IO

Current IO helpers include:

* save/load `HSI`
* save/load `Dictionary`
* save/load `TrainingSignals`
* load raw AVIRIS folders
* pack/unpack bitstreams

Generic HSI files are stored as `.npz`.

---

## AVIRIS Preprocessing

AVIRIS loading handles:

* `_img` / `_img.hdr` files
* ENVI interleave formats:

  * BIP
  * BIL
  * BSQ
* wavelengths from `.spc`
* scene metadata
* effective bit depth computed from data span

Additional preprocessing includes:

* filtering spectral bands
* removing water absorption bands
* enforcing increasing wavelengths without reordering bands
* trimming black borders
* cropping HSIs into fixed-size sections

---

## Design Decisions

Important decisions made so far:

1. Algorithms return domain objects, not raw tuples/dicts.
2. `HSIMetadata` is shared by raw and compressed HSIs.
3. `CompressedHSI.side_information` stores compressor-specific reconstruction data.
4. Compressors receive config names/specs for transforms and measurements, not prebuilt matrices.
5. Matrix construction lives in `transforms/`.
6. Tensor operations live in `math/`.
7. Bitstream packing lives in `io/`.
8. Metrics are plugin-style.
9. Runner computes metrics and creates final result objects.
10. UI owns the runner and receives status updates through callbacks.
11. Logging will consume completed result objects, not control execution.

---

## Example: Compression Run

```python
from src.pipeline.runner import Runner
from src.core.dictionary import Axis
from src.io.hsi import load_hsi
from src.compressors.hcs1d import HCS1D, HCS1DConfig
from src.metrics.compression import RMSE, PSNR, SAM, CompressionRate
from src.ui.console.status_view import ConsoleStatusView
from src.ui.console import result_view


hsi = load_hsi(r"data\raw\IndianPines.npz")

config = HCS1DConfig(
    K=3,
    sr=0.2,
    axis=Axis.SPECTRAL,
    Phi="SUBSAMPLING",
    Psi="IDCT",
)

compressor = HCS1D(config=config)

metrics = [
    RMSE(),
    PSNR(),
    SAM(),
    CompressionRate(),
]

status_view = ConsoleStatusView()
runner = Runner(status_callback=status_view.handle)

result_view.print_compression_run_header(hsi, compressor)

result = runner.run_compression(
    hsi=hsi,
    compressor=compressor,
    metrics=metrics,
)

result_view.print_compression_result(result)
```

---

## Example: Dictionary Training Run

```python
from src.pipeline.runner import Runner
from src.core.dictionary import Axis
from src.io.hsi import load_hsi
from src.preprocessing.training_signals import sample_training_signals
from src.dictionary_trainers.k_svd import K_SVD, K_SVDConfig
from src.metrics.dictionary import (
    RepresentationError,
    MeanSparsity,
    DictionaryCoherence,
)


hsi = load_hsi(r"data\raw\IndianPines.npz")

signals = sample_training_signals(
    hsi=hsi,
    num_signals=1000,
    axis=Axis.SPECTRAL,
    seed=0,
)

trainer = K_SVD(
    K_SVDConfig(
        K=64,
        T_0=3,
    )
)

metrics = [
    RepresentationError(),
    MeanSparsity(),
    DictionaryCoherence(),
]

runner = Runner()

result = runner.run_dictionary_training(
    signals=signals,
    trainer=trainer,
    metrics=metrics,
)
```

---

## Next Steps

Planned work:

1. Implement logging.
2. Add CSV or JSON summary logger.
3. Add optional artifact saving.
4. Clean up experiment scripts.
5. Add a full CLI.
6. Add BER sweep support.
7. Add more robust configuration loading.
8. Add tests for core data objects and runner behavior.

```
```
