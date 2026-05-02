project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
│
├── experiments/
│   ├── runs/
│   ├── configs/
│   └── logs/
│
├── src/
│   ├── core/
|   |   ├── hsi.py
|   |   └── experiment_item.py
│   ├── compressors/
|   |   ├── base_compressor.py
|   |   ├── ccsds123.py
|   |   ├── hcs1d.py
|   |   ├── hcs3d.py
|   |   └── cs/
|   |       ├── measurement_matrices.py
|   |       └── transforms.py
│   ├── dictionary/
|   |   └── k_svd.py
│   ├── math/
|   |   ├── n_way_ops.py
|   |   ├── metrics.py
|   |   └── recovery_algorithms.py
│   ├── io/
|   |   ├── loaders.py
|   |   ├── savers.py
|   |   └── aviris.py
│   ├── visualization/
|   |   ├── hsi_viewer.py
|   |   └── spectra.py
│   └── pipeline/
|       ├── experiment_runner.py
|       └── logger.py
│
├── interfaces/
│   ├── cli/
│   └── gui/
│
├── main.py
└── README.md