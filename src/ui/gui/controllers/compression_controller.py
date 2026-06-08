from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from src.compressors.registry import get_compressor, list_compressors
from src.pipeline.runner import Runner
from src import io
from src.loggers import ArtifactLoggerCallback, CSVLoggerCallback

from src.ui.gui.callbacks import GuiRunnerCallback


class CompressionController:
    """
    GUI-facing controller for compression workflows.
    """

    def __init__(self):
        pass

    def _create_runner(
        self,
        experiment_settings: dict,
        progress_callback=None,
        status_callback=None,
    ) -> Runner:
        results_dir = experiment_settings["results_dir"]

        callbacks = [
            ArtifactLoggerCallback(
                results_dir=results_dir,
                save_reconstructed=experiment_settings["save_reconstructed"],
                save_compressed=experiment_settings["save_compressed"],
                save_dictionary=False,
                save_coefficients=False,
                save_config=experiment_settings["save_config"],
                save_metadata=experiment_settings["save_metadata"],
            ),
            CSVLoggerCallback(
                results_dir=results_dir,
            ),
        ]

        if progress_callback is not None:
            callbacks.append(
                GuiRunnerCallback(
                    progress_callback=progress_callback,
                    status_callback=status_callback,
                )
            )

        return Runner(callbacks=callbacks)

    def available_compressors(self) -> list[str]:
        """
        Return all registered compressor names.
        """
        return list_compressors()

    def get_config_class(self, compressor_name: str):
        """
        Return the Config dataclass of a registered compressor.
        """
        compressor_cls = get_compressor(compressor_name)
        return compressor_cls.Config
    
    def get_config_fields(self, compressor_name: str):
        config_cls = self.get_config_class(compressor_name)

        if not is_dataclass(config_cls):
            raise TypeError(
                f"Config for compressor '{compressor_name}' must be a dataclass"
            )

        return list(fields(config_cls))

    def create_config(
            self,
            compressor_name: str,
            config_values: dict[str, Any],
        ):
            """
            Create a compressor Config object from GUI values.
            """
            config_cls = self.get_config_class(compressor_name)
            return config_cls(**config_values)

    def create_compressor(
        self,
        compressor_name: str,
        config_values: dict[str, Any],
        progress_callback=None,
    ):
        """
        Create a compressor object from GUI values.
        """
        compressor_cls = get_compressor(compressor_name)
        config = self.create_config(compressor_name, config_values)

        return compressor_cls(
            config=config,
            progress_callback=progress_callback,
        )

    def run_compression(
        self,
        hsi_path: Path,
        compressor_name: str,
        config_values: dict[str, Any],
        experiment_settings: dict[str, Any],
        progress_callback=None,
        status_callback=None,
    ) -> dict[str, Any]:
        hsi = self._load_hsi_from_path(hsi_path)

        compressor = self.create_compressor(
            compressor_name=compressor_name,
            config_values=config_values,
        )

        runner = self._create_runner(
            experiment_settings=experiment_settings,
            progress_callback=progress_callback,
            status_callback=status_callback,
        )

        result = runner.run_compression(
            hsi=hsi,
            compressor=compressor,
            experiment=experiment_settings["experiment"],
            ber=experiment_settings["ber"],
        )

        return self._format_result(result)
    
    def _load_hsi_from_path(self, hsi_path: Path):
        """
        Load an HSI object from a selected GUI path.
        """
        folder = hsi_path.parent
        name = hsi_path.stem
        return io.load_hsi(folder, name)

    def _format_result(self, result) -> dict[str, Any]:
        """
        Convert CompressionRunResult into GUI-friendly values.
        """
        output = {}

        for name, metric in result.metrics.items():
            value = getattr(metric, "value", metric)
            output[name] = value

        return output