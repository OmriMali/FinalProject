# GUI Module Architecture

## MainWindow

`main_window.py` is the GUI shell. It creates the shared workspace panel,
the compression tab, the results tab, and connects high-level signals.

It should not contain plotting logic, compression process logic, or artifact
loading logic.

## Widgets

`widgets/` contains reusable Qt widgets:

- `WorkspacePanel`: file controls and loaded-items table
- `CompressionTab`: experiment settings, compressor settings, run controls
- `ResultsTab`: visualization controls, metrics table, and Matplotlib canvas
- `MetricsTableWidget`: metrics display table

## Controllers

`controllers/` contains GUI workflow logic:

- `CompressionController`: QProcess lifecycle and JSON message parsing
- `VisualizationController`: object loading and visualization dispatch
- `ArtifactController`: loading artifacts returned by compression runs

## Services

`services/` contains lower-level helpers:

- `WorkspaceLoader`: loads and inspects HSI and CompressedHSI files
- `metrics_extractor`: converts stored metric data into display objects

## Processes

`processes/` contains subprocess entry points, such as the compression job
runner used by `CompressionController`.