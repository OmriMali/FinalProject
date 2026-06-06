from src import io, data_processing, visuals
import matplotlib.pyplot as plt


def main():

    # hybrid_path = r"results\compression\hybrid\log.csv"
    ccsds123_path = r"results\compression\ccsds123\log.csv"
    # ccsds_path = r"results\compression\ccsds123\log.csv"

    df = data_processing.load_logs([ccsds123_path])
    df = data_processing.filter_by(df, experiment="ccsds_local_sum_mode_sweep")

    cr_vs_local_sum_mode = data_processing.aggregate_mean_std(
        df,
        group_cols=["method","local_sum_mode"],
        value_cols=["cr"]
    )

    # times = data_processing.aggregate_mean_std(
    #     df,
    #     group_cols=["method"],
    #     value_cols=["comp_time", "decomp_time"]
    # )

    visuals.plot_metric_vs_metric(
        cr_vs_local_sum_mode,
        x="local_sum_mode",
        xlabel="Local Sum Mode",
        y="cr_mean",
        ylabel="Compression Ratio",
        title="Compression Ratio vs Local Sum Mode",
        # show_legend=False,
        style=visuals.DEFAULT_STYLE,
        plot_type="bar"
    )

    # visuals.plot_runtime_comparison(
    #     times,
    #     compression_error_col="comp_time_std",
    #     decompression_error_col="decomp_time_std",
    #     style=visuals.DARK_STYLE
    # )

    plt.show()


if __name__ == "__main__":
    main()