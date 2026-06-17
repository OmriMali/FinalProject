from src import io, compressors, visuals, metrics
from src.core.dictionary import Axis
from src.pipeline.runner import Runner
from src.ui.console.callback import ConsoleCallback
from src.loggers.csv import CSVLoggerCallback
from src.loggers.artifacts import ArtifactLoggerCallback
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from src.io import logs
from src.data_processing import filters, aggregate
from src.visuals import metrics





def main():
    hsi_path = r"data\sections\JasperRidge"
    hsi_name = "JasperRidge_r6_c2"
    hsi = io.load_hsi(directory=hsi_path, name=hsi_name)
    hsi_rec_path = r"C:\Users\omrim\Documents\FinalProject\results\compression\hybrid\artifacts\jasper_ridge_r6_c2_hybrid_visual_test_20260612_123127"
    hsi_rec_name = "reconstructed"
    hsi_rec = io.load_hsi(directory=hsi_rec_path, name=hsi_rec_name)
    diff = hsi.data.astype(float) - hsi_rec.data.astype(float)
    value = np.sqrt(np.mean(diff ** 2))
    print(value)
    # hsi_rec=io.load_hsi(r"results\compression\hcs1d\artifacts\jasper_ridge_r1_c2_hcs1d_test_20260526_065545", name="reconstructed")
    figure, axes = visuals.compare_rgb(hsis = [hsi], labels = ['hsi'])
    plt.show()
if __name__ == "__main__":
    main()
