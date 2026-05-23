from itertools import product

from src import io, compressors
from src.core.dictionary import Axis
from src.pipeline.runner import Runner
from src.ui.console.callback import ConsoleCallback
from src.loggers.csv import CSVLoggerCallback
from src.loggers.artifacts import ArtifactLoggerCallback


def sweep_configs(config_cls, fixed: dict, sweep: dict):
    """
    Generate config objects from fixed parameters and swept parameters.
    """

    keys = list(sweep.keys())
    values = list(sweep.values())

    for combo in product(*values):
        params = fixed.copy()
        params.update(dict(zip(keys, combo)))
        yield config_cls(**params)


def main():

    # ===== Run info =====
    compressors_to_run = [
        "hcs1d",
        # "hcs3d",
        "ccsds123",
    ]
    
    save_reconstructed = False
    save_compressed = False

    tags = {"experiment_group": "poster_1"}

    # ===== Paths =====
    hsi_dir = r"data\processed\JasperRidge\size_sweep"
    hsi_name = "JasperRidge_h_128_w_128"

    dict_dir = r"results\artifacts\dictionary\2026-05-16T16_18_40_ksvd_ksvd"
    dict_name = "dictionary"
    learned_base = f"LEARNED:directory={dict_dir},name={dict_name}"

    artifacts_dir = r"results/artifacts"
    log_dir = r"results/logs"

    # ===== Load HSI =====
    hsi = io.load_hsi(hsi_dir, hsi_name)

    # ===== Pipeline ===== 
    runner = Runner(
        callbacks=[
            ArtifactLoggerCallback(
                root_dir=artifacts_dir,
                save_reconstructed=save_reconstructed,
                save_compressed=save_compressed,
            ),
            CSVLoggerCallback(log_dir=log_dir),
            ConsoleCallback(),
        ]
    )

    # ===== Experiment configs =====
    experiments = {
        "hcs1d": {
            "config_cls": compressors.HCS1DConfig,
            "fixed": {
                "axis": Axis.SPECTRAL,
                "Phi": "SUBSAMPLING",
                "Psi": learned_base,
                "K": 3
            },
            "sweep": {
                "sr": [1/5, 1/8, 1/10, 1/15],
            },
        },

        "hcs3d": {
            "config_cls": compressors.HCS3DConfig,
            "fixed": {
                "Phis": ("SUBSAMPLING", "SUBSAMPLING", "SUBSAMPLING"),
                "Psis": ("IDCT", "IDCT", learned_base),
            },
            "sweep": {
                "K": [2000, 3000, 4000, 5000],
                "sr": [
                    (0.5, 0.5, 0.05),
                    (0.5, 0.5, 0.1),
                    (0.8, 0.8, 0.1),
                ],
            },
        },

        "ccsds123": {
            "config_cls": compressors.CCSDS123Config,
            "fixed": {
                "local_sum_mode": "column",
                "P": 2,
                "Omega": 8,
                "block_size": 32,
            },
            "sweep": {
                "a": [10, 40, 100, 400],
            },
        },
    }

    for compressor_name in compressors_to_run:
        spec = experiments[compressor_name]
        compressor_cls = compressors.registry.get_compressor(compressor_name)

        for config in sweep_configs(
            config_cls=spec["config_cls"],
            fixed=spec["fixed"],
            sweep=spec["sweep"],
        ):
            compressor = compressor_cls(config)

            runner.run_compression(
                hsi=hsi,
                compressor=compressor,
                tags=tags
            )


if __name__ == "__main__":
    main()