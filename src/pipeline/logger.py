import csv
import json
from pathlib import Path

from src.core.compression_run_item import CompressionRunItem
from src.core.dictionary_learning_item import DictionaryLearningItem
from src.io.savers import save_hsi, save_array_to_path

class Logger:
    def __init__(self, root_dir="results"):
        self.root_dir = Path(root_dir)

        self.comp_dir = self.root_dir / "compression"
        self.comp_runs = self.comp_dir / "runs"
        self.comp_csv = self.comp_dir / "runs.csv"

        self.dict_dir = self.root_dir / "dictionaries"
        self.dict_runs = self.dict_dir / "runs"
        self.dict_csv = self.dict_dir / "runs.csv"

        self.comp_dir.mkdir(parents=True, exist_ok=True)
        self.dict_dir.mkdir(parents=True, exist_ok=True)
        self.comp_runs.mkdir(parents=True, exist_ok=True)
        self.dict_runs.mkdir(parents=True, exist_ok=True)


    def log_compression(self, item: CompressionRunItem):
        exp_path = self._create_comp_folder(item)

        self._save_config(exp_path, item)
        self._save_metrics(exp_path, item)
        self._save_reconstruction(exp_path, item)

        self._append_csv_comp(item)

        return exp_path
    
    def _create_comp_folder(self, item: CompressionRunItem):
        exp_path = self.comp_runs / item.run_id
        exp_path.mkdir(parents=True, exist_ok=True)
        item.output_dir = exp_path
        return exp_path

    def _save_config(self, exp_path, item: CompressionRunItem):
        config = {
            "run_id": item.run_id,
            "timestamp": item.timestamp,
            "machine": item.experiment_machine,
            "ber": item.ber,
            "tag": item.tag,
            "save_hsi": item.save_hsi,

            "hsi": {
                "sensor": item.hsi.metadata.get("sensor"),
                "site": item.hsi.metadata.get("site"),
                "name": item.hsi.metadata.get("name"),
            },

            "compressor": {
                "name": item.compressor_name,
                "params": item.compressor_params
            }
        }

        with open(exp_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def _save_reconstruction(self, exp_path, item: CompressionRunItem):
        if not item.save_hsi:
            return
        
        save_hsi(item.reconstructed, exp_path / "reconstruction.npy")

    def _append_csv_comp(self, item: CompressionRunItem):
        
        row = {
            # experiment
            "run_id": item.run_id,
            "timestamp": item.timestamp,
            "machine": item.experiment_machine,
            "hsi_exists": item.save_hsi,
            "ber": item.ber,
            "tag": item.tag,

            # HSI
            "sensor": item.hsi.metadata.get("sensor"),
            "site": item.hsi.metadata.get("site"),
            "name": item.hsi.metadata.get("name"),

            # compressor
            "compressor_name": item.compressor_name,

            # metrics
            **item.metrics
        }

        file_exists = self.comp_csv.exists()

        with open(self.comp_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)


    def log_dictionary(self, item: DictionaryLearningItem):
        exp_path = self._create_dict_folder(item)

        self._save_dict_config(exp_path, item)
        self._save_metrics(exp_path, item)
        self._save_dictionary(exp_path, item)

        self._append_csv_dictionary(item)
        
        return exp_path
    
    def _create_dict_folder(self, item: DictionaryLearningItem):
        exp_path = self.dict_runs / item.run_id
        exp_path.mkdir(parents=True, exist_ok=True)
        item.output_dir = exp_path
        return exp_path
    
    def _save_dict_config(self, exp_path, item: DictionaryLearningItem):
        config = {
            "run_id": item.run_id,
            "timestamp": item.timestamp,
            "machine": item.experiment_machine,
            "tag": item.tag,

            "dict_name": item.dict_name,
            
            "algorithm_name": item.algorithm_name,
            "algorithm_params": item.algorithm_params,
        }

        with open(exp_path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    def _save_dictionary(self, exp_path, item: DictionaryLearningItem):
        if item.D is None:
            return
        
        save_array_to_path(item.D, exp_path / f"{item.dict_name}.npz")

    def _append_csv_dictionary(self, item: DictionaryLearningItem):
        row = {
            "run_id": item.run_id,
            "timestamp": item.timestamp,
            "machine": item.experiment_machine,
            "tag": item.tag,

            "dict_name": item.dict_name,
            "algorithm": item.algorithm_name,

            **item.metrics
        }

        file_exists = self.dict_csv.exists()

        with open(self.dict_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)


    def _save_metrics(self, exp_path, item):
        with open(exp_path / "metrics.json", "w") as f:
            json.dump(item.metrics, f, indent=2)


