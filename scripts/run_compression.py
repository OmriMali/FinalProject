from src import io, compressors
from src.core.dictionary import Axis
from src.pipeline.runner import Runner
from src.ui.console.callback import ConsoleCallback
from src.loggers.csv import CSVLoggerCallback
from src.loggers.artifacts import ArtifactLoggerCallback


def main():

    # ===== Run info =====
    compressor_name = "ccsds123"           # "hcs1d", "hcs3d", "ccsds123", "hybrid"
    experiment = "ccsds123_test"
    ber = 0.000001
    tags = None

    save_reconstructed = True
    save_compressed = False

    # ===== Paths =====
    hsi_dir = r"data\sections\JasperRidge"
    hsi_name = "JasperRidge_r6_c2"

    dict_dir = r"resources\dictionaries"
    dict_name = r"jasper_ridge_split_01_ksvd_400_atoms.npz"
    learned_base = f"LEARNED:directory={dict_dir},name={dict_name}"

    results_dir = r"results"

    # ===== Configs =====
    configs = {
        "hcs1d": compressors.HCS1DConfig(
            K=3,
            sr=0.05,
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
            a=400,
            block_size=32,
        ),
        "hybrid": compressors.HybridConfig(
            K=3,
            sr=0.05,
            Phi="GAUSSIAN",
            Psi=learned_base,
            block_size=32,
            protect_bitstream=False
        )
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

    runner.run_compression(hsi, compressor, experiment, ber=ber, tags=tags)

    

if __name__ == "__main__":
    main()