from src import io, data_processing, visuals
import matplotlib.pyplot as plt


def main():

    hybrid_path = r"results\compression\hybrid\log.csv"
    hcs1d_path = r"results\compression\hcs1d\log.csv"
    ccsds_path = r"results\compression\ccsds123\log.csv"

    df = data_processing.load_logs([hybrid_path, hcs1d_path, ccsds_path])
    df = data_processing.filter_by(df, experiment="compressors_comparison")

    rmse_vs_cr = data_processing.aggregate_mean_std(
        df,
        group_cols=["method","sr","a"],
        value_cols=["rmse", "cr"]
    )

    times = data_processing.aggregate_mean_std(
        df,
        group_cols=["method"],
        value_cols=["comp_time", "decomp_time"]
    )

    visuals.plot_metric_vs_metric(
        rmse_vs_cr,
        method_col="method",
        x="cr_mean",
        xerr="cr_std",
        xlabel="cr",
        y="rmse_mean",
        yerr="rmse_std",
        ylabel="RMSE",
        title="RMSE vs Compression Ratio",
        # show_legend=False,
        style=visuals.DEFAULT_STYLE,
        # plot_type="bar"
    )

    visuals.plot_runtime_comparison(
        times,
        method_col="method",
        compression_error_col="comp_time_std",
        decompression_error_col="decomp_time_std",
        style=visuals.DEFAULT_STYLE
    )

    plt.show()


if __name__ == "__main__":
    main()