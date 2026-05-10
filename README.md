# Hyperspectral Image Compression Framework

This project is a lightweight research framework for testing hyperspectral image compression algorithms, dictionary learning methods, and reconstruction quality metrics.

The main goal is to keep the code modular while avoiding overengineering. The framework separates:

- core data objects
- compression algorithms
- dictionary learning algorithms
- metrics
- runners
- IO
- preprocessing
- UI / console output

---

## Project Structure

```text
src/
├── core/
│   ├── hsi.py
│   ├── dictionary.py
│   ├── training_signals.py
│   └── results.py
│
├── compressors/
│   ├── base.py
│   ├── registry.py
│   ├── hcs1d.py
│   ├── hcs3d.py
│   └── ccsds123.py
│
├── dictionary_trainers/
│   ├── base.py
│   ├── registry.py
│   └── k_svd.py
│
├── metrics/
│   ├── base.py
│   ├── registry.py
│   ├── compression.py
│   └── dictionary.py
│
├── pipeline/
│   ├── runner.py
│   └── status.py
│
├── io/
│   ├── hsi.py
│   ├── aviris.py
│   ├── dictionary.py
│   ├── training_signals.py
│   └── bitstream.py
│
├── preprocessing/
│   ├── hsi.py
│   └── training_signals.py
│
├── math/
│   ├── numeric.py
│   ├── n_way_ops.py
│   └── regression_algs.py
│
├── transforms/
│   ├── measurements.py
│   └── sparse_bases.py
│
└── ui/
    └── console/
        ├── status_view.py
        └── result_view.py