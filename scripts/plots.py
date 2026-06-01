from src import io, data_processing, visuals
import matplotlib.pyplot as plt


def main():

    hybrid_path = r"results\compression\hybrid\log.csv"
    hcs1d_path = r"results\compression\hcs1d\log.csv"
    ccsds_path = r"results\compression\ccsds123\log.csv"

    df = data_processing.load_logs([hybrid_path, hcs1d_path, ccsds_path])
    df = data_processing.filter_by(df, experiment="comparison_1")

    rmse_v_cr = data_processing.aggregate_mean_std(
        df,
        group_cols=["method", "sr", "a"],
        value_cols=["rmse", "cr"]
    )

    times = data_processing.aggregate_mean_std(
        df,
        group_cols=["method"],
        value_cols=["comp_time", "decomp_time"]
    )

    visuals.plot_metric_vs_metric(
        rmse_v_cr,
        x="cr_mean",
        xlabel="Compression Rate",
        y="rmse_mean",
        yerr="rmse_std",
        ylabel="RMSE",
        style=visuals.DARK_STYLE
    )

    visuals.plot_runtime_comparison(
        times,
        compression_error_col="comp_time_std",
        decompression_error_col="decomp_time_std",
        style=visuals.DARK_STYLE
    )

    plt.show()


if __name__ == "__main__":
    main()