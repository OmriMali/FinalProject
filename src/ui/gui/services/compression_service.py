from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

from src.compressors.registry import get_compressor
from src.core.hsi import HSI
from src.loggers import ArtifactLoggerCallback, CSVLoggerCallback
from src.pipeline.runner import Runner
from src.ui.gui.models.workspace_item import WorkspaceItem
from src.ui.gui.callbacks import GuiRunnerCallback


@dataclass
class CompressionGuiResult:
    """
    GUI-facing compression result.
    """

    source_item: WorkspaceItem
    compressor_name: str
    result: Any
    artifact_dir: Path | None = None
    artifact_paths: dict[str, Path] | None = None

class CompressionService:
    """
    Run compression workflows for the GUI.

    MainWindow should not know how compressors, runners, or loggers are
    constructed.
    """

    def compress_and_decompress(
        self,
        hsi: HSI,
        source_item: WorkspaceItem,
        compressor_name: str,
        config_values: dict[str, Any],
        experiment_settings: dict[str, Any],
        progress_callback: Callable[[float], None] | None = None,
        message_callback: Callable[[str], None] | None = None,
    ) -> CompressionGuiResult:
        compressor = self._create_compressor(
            compressor_name=compressor_name,
            config_values=config_values,
            progress_callback=progress_callback,
        )

        runner, artifact_callback = self._create_runner(
            experiment_settings=experiment_settings,
            progress_callback=progress_callback,
            message_callback=message_callback,
        )

        result = runner.run_compression(
            hsi=hsi,
            compressor=compressor,
            experiment=experiment_settings["experiment"],
            ber=experiment_settings["ber"],
        )

        artifact_dir = (
            artifact_callback.last_artifact_dir
            if artifact_callback is not None
            else None
        )
        artifact_paths = (
            artifact_callback.last_artifact_paths
            if artifact_callback is not None
            else {}
        )

        return CompressionGuiResult(
            source_item=source_item,
            compressor_name=compressor_name,
            result=result,
            artifact_dir=artifact_dir,
            artifact_paths=artifact_paths,
        )

    def _create_compressor(
        self,
        compressor_name: str,
        config_values: dict[str, Any],
        progress_callback: Callable[[float], None] | None = None,
    ):
        compressor_cls = get_compressor(compressor_name)
        config_cls = compressor_cls.Config

        config = self._create_config(config_cls, config_values)

        return compressor_cls(
            config=config,
            progress_callback=progress_callback,
        )

    def _create_config(
        self,
        config_cls,
        config_values: dict[str, Any],
    ):
        if not is_dataclass(config_cls):
            raise TypeError("Compressor Config must be a dataclass")

        valid_fields = {field.name for field in fields(config_cls)}

        filtered_values = {
            key: value
            for key, value in config_values.items()
            if key in valid_fields
        }

        return config_cls(**filtered_values)

    def _create_runner(
        self,
        experiment_settings: dict[str, Any],
        progress_callback: Callable[[float], None] | None = None,
        message_callback: Callable[[str], None] | None = None,
    ):
        results_dir = experiment_settings["results_dir"]
        save_result = experiment_settings.get("save_result", True)

        artifact_callback = None
        callbacks = []

        if save_result:
            artifact_callback = ArtifactLoggerCallback(
                results_dir=results_dir,
                save_reconstructed=True,
                save_compressed=True,
                save_dictionary=False,
                save_coefficients=False,
                save_config=True,
                save_metadata=True,
            )

            callbacks.extend(
                [
                    artifact_callback,
                ]
            )

        callbacks.append(
            CSVLoggerCallback(
                results_dir=results_dir,
            )
        )

        if progress_callback is not None or message_callback is not None:
            callbacks.append(
                GuiRunnerCallback(
                    progress_callback=progress_callback,
                    message_callback=message_callback,
                )
            )

        runner = Runner(callbacks=callbacks)

        return runner, artifact_callback
