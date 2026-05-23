from src import io, compressors
from src.core.dictionary import Axis
from src.pipeline.runner import Runner
from src.ui.console.callback import ConsoleCallback
from src.loggers.csv import CSVLoggerCallback
from src.loggers.artifacts import ArtifactLoggerCallback


def main():

    # ===== Run info =====
    compressor_name = "hcs1d"           # "hcs1d", "hcs3d", "ccsds123"
    experiment = "test"
    ber = 0.0
    tags = None

    save_reconstructed = True
    save_compressed = False

    # ===== Paths =====
    hsi_dir = r"data\processed\JasperRidge\sections"
    hsi_name = "JasperRidge_r1_c1"

    dict_dir = r"results\dictionary_training\ksvd\artifacts\ksvd\ksvd_test_20260524_000209"
    dict_name = r"dictionary"
    learned_base = f"LEARNED:directory={dict_dir},name={dict_name}"

    results_dir = r"results"

    # ===== Configs =====
    configs = {
        "hcs1d": compressors.HCS1DConfig(
            K=3,
            sr=0.1,
            axis=Axis.SPECTRAL,
            Phi="SUBSAMPLING",
            Psi=learned_base,
        ),
        "hcs3d": compressors.HCS3DConfig(
            K=10000,
            sr=(0.5, 0.5, 0.1),
            Phis=("SUBSAMPLING", "SUBSAMPLING", "SUBSAMPLING"),
            Psis=("IDCT", "IDCT", learned_base),
        ),
        "ccsds123": compressors.CCSDS123Config(
            local_sum_mode="column",
            P=2,
            Omega=8,
            a=100,
            block_size=32,
        ),
    }

    # ===== Load HSI =====
    hsi = io.load_hsi(hsi_dir, hsi_name)

    # ===== Build compressor =====
    obj = compressors.registry.get_compressor(compressor_name)
    compressor = obj(configs[compressor_name])

    # ===== Pipeline =====
    runner = Runner(
        callbacks=[
            ArtifactLoggerCallback(
                results_dir=results_dir,
                save_reconstructed=save_reconstructed,
                save_compressed=save_compressed,
                save_config=True,
            ),
            CSVLoggerCallback(
                results_dir=results_dir
            ),
            ConsoleCallback()
        ]
    )

    runner.run_compression(hsi, compressor, experiment, ber, tags=tags)

    

if __name__ == "__main__":
    main()