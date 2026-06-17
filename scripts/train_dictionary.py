from src import io, dictionary_trainers
from src.pipeline.runner import Runner
from src.ui.console.callback import ConsoleCallback
from src.loggers.csv import CSVLoggerCallback
from src.loggers.artifacts import ArtifactLoggerCallback


def main():

    # ===== Run info ====
    trainer_name = "ksvd"           # "ksvd"
<<<<<<< HEAD
    experiment = "test2"
=======
    experiment = "test1"
>>>>>>> master

    save_dictionary = True
    save_coefficients = True

    # ===== Paths =====
    signals_dir = r"data\training_signals"
<<<<<<< HEAD
    signals_name = "mosfet_field_split_01_diverse_spectral"
=======
    signals_name = "cuprite_split_01_diverse_spectral"
>>>>>>> master

    results_dir = "results"

    # ===== Configs =====
    configs = {
        "ksvd": dictionary_trainers.K_SVDConfig(
<<<<<<< HEAD
        K=400,
=======
        K=380,
>>>>>>> master
        T_0=3,
        tol=0.01,
        max_iter=50
        )
    }

    # ===== Load signals =====
    signals = io.load_training_signals(signals_dir, signals_name)
    
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