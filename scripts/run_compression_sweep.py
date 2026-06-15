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
        "hybrid",
        # "ccsds123",
    ]
    experiment = "compressors_comparison"
    ber = 0
    tags = None
    
    save_reconstructed = False
    save_compressed = False

    use_split = False
    repetitions = {"ccsds123": 1, "hcs1d": 10, "hybrid": 10}

    # ===== Paths =====
    split_csv = r"resources\splits\jasper_ridge_split_01.csv"
    hsi_dir = r"data\sections\JasperRidge"
    hsi_name = "JasperRidge_r11_c1"
    
    dict_dir = r"resources\dictionaries"
    dict_name = "jasper_ridge_split_01_ksvd_400_atoms.npz"

    results_dir = r"results"
    protect_bitstream=ber>0
    # ===== Load HSIs =====
    test_hsis = []
    if use_split:
        test_hsis = io.load_hsi_split(split_csv, "test")
    else:
        test_hsis.append(io.load_hsi(hsi_dir, hsi_name))

    # ===== Pipeline ===== 
    runner = Runner(
        callbacks=[
            ArtifactLoggerCallback(
                results_dir=results_dir,
                save_reconstructed=save_reconstructed,
                save_compressed=save_compressed,
                save_config=True,
                save_metadata=False
            ),
            CSVLoggerCallback(results_dir=results_dir),
            ConsoleCallback(),
        ]
    )

    # ===== Experiment configs =====
    learned_base = f"LEARNED:directory={dict_dir},name={dict_name}"

    experiments = {
        "hcs1d": {
            "config_cls": compressors.HCS1DConfig,
            "fixed": {
                "axis": Axis.SPECTRAL,
                "Phi": "BERNOULLI",
                "Psi": learned_base,
                "K": 3
            },
            "sweep": {
                "sr": [1/2, 1/4, 1/8, 1/16, 1/32],
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
                "P": 2,
                "Omega": 8,
                "block_size": 32,
                "local_sum_mode": "neighbor",
                "protect_bitstream": protect_bitstream,
            },
            "sweep": {
                "a": [4, 10, 40, 100, 400],
            },
        },

        "hybrid": {
            "config_cls": compressors.HybridConfig,
            "fixed": {
                "K": 3,
                "Phi": "BERNOULLI",
                "Psi":learned_base,
                "local_sum_mode":"neighbor",
                "block_size": 32,
                "protect_bitstream": protect_bitstream,
            },
            "sweep": {
                "sr": [(1/2)*1.3, (1/4)*1.3, (1/8)*1.3, (1/16)*1.3, (1/32)*1.3],
                # "local_sum_mode": ["hybrid_mean", "column", "neighbor"]
                # "Phi": ["BERNOULLI:p=0.1", "BERNOULLI:p=0.25", "BERNOULLI:p=0.5"]
            }
        }
    }

    # ===== Loop over HSIs =====
    for hsi in test_hsis:
        # ===== Loop over compressor =====
        for compressor_name in compressors_to_run:
            # ===== Loop over repetitions =====
            for _ in range(repetitions[compressor_name]):
                spec = experiments[compressor_name]
                compressor_cls = compressors.registry.get_compressor(compressor_name)
                # ===== Loop over sweep parameters =====
                for config in sweep_configs(
                    config_cls=spec["config_cls"],
                    fixed=spec["fixed"],
                    sweep=spec["sweep"],
                ):
                    compressor = compressor_cls(config)
                    runner.run_compression(
                        hsi=hsi,
                        compressor=compressor,
                        experiment=experiment,
                        ber=ber,
                        tags=tags
                    )

if __name__ == "__main__":
    main()