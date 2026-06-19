# GUI Module Architecture

## MainWindow

`main_window.py` is the GUI shell. It creates the shared workspace panel,
the compression, visualization, and data-analysis tabs, and connects
high-level signals.

It should not contain plotting logic, compression process logic, or artifact
loading logic.

## Widgets

`widgets/` contains reusable Qt widgets:

- `WorkspacePanel`: file controls and loaded-items table
- `CompressionTab`: experiment settings, compressor settings, run controls, and run metrics
- `VisualizationTab`: HSI visualization controls and Matplotlib canvas
- `DataAnalysisTab`: log loading, filtering, tables, and metric plots
- `MetricsTableWidget`: metrics display table
- `FigurePopoutWindow`: standalone Matplotlib figure viewer

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
