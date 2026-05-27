# Compressive Sensing for Hyperspectral Images

A toolbox for testing hyperspectral image compressors, with a focus
on compressive sensing based compressors.

---

## To Do

1. Add a parser for objects like compressor configs to normalize printing\writing.
2. Think of a way for the UI to know about sweeps, and show sweep progress accordingly.
3. Add a GUI
4. Document this repo...

---

## Directory Structure

```text
project/
├── data/
│   ├── raw/
│   │   └── <dataset>/
│   │       └── Raw AVIRIS / hyperspectral files
│   │
│   ├── processed/
│   │   └── <dataset>/
│   │       └── <dataset>.npz
│   │
│   ├── sections/
│   │   └── <dataset>/
│   │       ├── <dataset>_r1_c1.npz
│   │       ├── <dataset>_r1_c2.npz
│   │       └── ...
│   │
│   └── training_signals/
│       └── *.npz
│
├── resources/
│   ├── dictionaries/
│   │   └── Shared tracked dictionaries
│   │
│   └── splits/
│       └── Train/test split CSV files
│
├── results/
│   ├── compression/
│   │   └── compressor/
│   │       ├── artifacts/
|   |       │   └── <scene>_<experiment>_<timestamp>/
│   │       │        ├── reconstructed.npz
│   │       │        ├── compressed.npz
│   │       │        └── config.json
|   |       │
│   │       └── log.csv
│   │
│   └── dictionary_training/
│       └── algorithm/
│           ├── artifacts/
|           │   └── <algorithm>_<experiment>_<timestamp>/
│           │        ├── dictionary.npz
│           │        ├── coefficients.npz
│           │        └── config.json
│           │
│           └── log.csv
│
├── scripts/
│   ├── preprocess_aviris.py
│   ├── train_test_split.py
│   ├── create_training_signals.py
│   ├── train_dictionary.py
│   ├── run_compression.py
│   └── ...
│
├── src/
│   ├── compressors/
│   ├── core/
│   ├── data_processing/
│   ├── dictionary_trainers/
│   ├── io/
│   ├── loggers/
│   ├── metrics/
│   ├── pipeline/
│   ├── preprocessing/
│   ├── transforms/
│   ├── ui/
│   ├── utils/
│   └── visuals/
│
├── README.md
├── requirements.txt
└── .gitignore
```