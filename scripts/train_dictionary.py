from src import io, dictionary_trainers
from src.pipeline.runner import Runner
from src.ui.console.callback import ConsoleCallback
from src.loggers.csv import CSVLoggerCallback
from src.loggers.artifacts import ArtifactLoggerCallback


def main():

    # ===== Run info ====
    trainer_name = "ksvd"           # "ksvd"
    experiment = "test"

    save_dictionary = True
    save_coefficients = False

    # ===== Paths =====
    # signals_dir = r"data\training"
    signals_dir = r"data\processed\train"
    signals_name = "sampled_JasperRidge_r1_c1"

    results_dir = "results"

    # ===== Configs =====
    configs = {
        "ksvd": dictionary_trainers.K_SVDConfig(
        K=400,
        T_0=3,
        tol=0.01,
        max_iter=50
        )
    }

    # ===== Load signals =====
    # signals = io.load_training_signals(signals_dir, signals_name)
    signals = io.load_diverse_signals_library(
        folder_path=signals_dir,
        name=signals_name,
        threshold=0.995,
        max_atoms=configs[trainer_name].K * 10,
        seed=42
    )

    # ===== Build trainer =====
    obj = dictionary_trainers.registry.get_dictionary_trainer(trainer_name)
    trainer = obj(configs[trainer_name])

    # ===== Pipeline =====
    runner = Runner(
        callbacks=[
            ArtifactLoggerCallback(
                results_dir=results_dir,
                save_dictionary=save_dictionary,
                save_coefficients=save_coefficients,
            ),
            CSVLoggerCallback(
                results_dir=results_dir
            ),
            ConsoleCallback()
        ]
    )
    runner.run_dictionary_training(signals, trainer, experiment=experiment)


if __name__ == "__main__":
    main()