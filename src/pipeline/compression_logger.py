import numpy as np
import os
import csv
import json
from pathlib import Path

from src.core.experiment_item import ExperimentItem


class CompressionLogger:

    def __init__(self, root_dir="results"):
        self.root_dir = Path(root_dir)
        self.csv_path = self.root_dir / "experiments.csv"
        self.exp_dir = self.root_dir / "experiments"

        self.exp_dir.mkdir(parents=True, exist_ok=True)


    def log(self, item: ExperimentItem):
        exp_path = self._create_exp_folder(item)

        self._save_config(exp_path, item)
        self._save_metrics(exp_path, item)
        self._save_reconstruction(exp_path, item)

        self._append_csv(item, exp_path)

        return exp_path
    
    def _create_exp_folder(self, item: ExperimentItem):
        exp_path = self.exp_dir / item.experiment_id
        exp_path.mkdir(parents=True, exist_ok=True)
        item.output_dir = exp_path
        return exp_path
    
    def _save_config(self, exp_path, item: ExperimentItem):
        config = {
            "experiment_id": item.experiment_id,
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

    def _save_metrics(self, exp_path, item: ExperimentItem):
        with open(exp_path / "metrics.json", "w") as f:
            json.dump(item.metrics, f, indent=2)

    def _save_reconstruction(self, exp_path, item: ExperimentItem):
        if not item.save_hsi:
            return
        
        np.save(exp_path / "reconstruction.npy", item.reconstructed)

    def _append_csv(self, item: ExperimentItem, exp_path):
        
        row = {
            # experiment
            "experiment_id": item.experiment_id,
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

        file_exists = self.csv_path.exists()

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(row)