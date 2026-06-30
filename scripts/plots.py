from src import io, data_processing, visuals
import matplotlib.pyplot as plt


def main():

    hybrid_path = r"results\compression\hybrid\log.csv"
    hcs1d_path = r"results\compression\hcs1d\log.csv"
    ccsds_path = r"results\compression\ccsds123\log.csv"

    df = data_processing.load_logs([hcs1d_path, hybrid_path, ccsds_path])
    df = data_processing.filter_by(df, experiment="project_book_results_comparison")
    df = data_processing.filter_compare(df, "sr", op="!=", value=0.05)

    # rmse_v_sr = data_processing.aggregate_mean_std(
    #     df,
    #     group_cols=["method","sr", "a"],
    #     value_cols=["cr", "rmse", "sam", "comp_time", "decomp_time"]
    # )
    # data_processing.save_dataframe_to_csv(rmse_v_sr,
    #                                      r"resources\tables\project_book_results_comparison_aggregated.csv")

    # times = data_processing.aggregate_mean_std(
    #     df,
    #     group_cols=["method"],
    #     value_cols=["comp_time", "decomp_time"]
    # )


    # visuals.plot_metric_vs_metric(
    #     rmse_v_sr,
    #     method_col="method",
    #     x="cr_mean",
    #     xerr="cr_std",
    #     xlabel="Compression Ratio",
    #     y="rmse_mean",
    #     yerr="rmse_std",
    #     ylabel="Root Mean Squared Error",
    #     style=visuals.DEFAULT_STYLE,
    # )

    # visuals.plot_runtime_comparison(
    #     times,
    #     method_col="method",
    #     compression_error_col="comp_time_std",
    #     decompression_error_col="decomp_time_std",
    #     style=visuals.DEFAULT_STYLE
    # )

    original = io.load_hsi(r"data\sections\JasperRidge", "JasperRidge_r6_c2.npz")

    has_rec = data_processing.filter_has_reconstructed_hsi(df)
    has_rec = data_processing.keep_columns(has_rec, ["run_id", "method", "cr", "rmse", "sam", "artifact_dir"])
    # data_processing.save_dataframe_to_csv(has_rec,
    #                                        r"resources\tables\project_book_reconstruction_map.csv")

    cr_8_map = [
        {"run_id": 31, "method": "ccsds123"},
        {"run_id": 449, "method": "hcs1d"},
        {"run_id": 3, "method": "hybrid"},
    ]
    cr_8_hsis = data_processing.load_reconstructed_hsis(has_rec, criteria=cr_8_map)

    # visuals.compare_rgb(cr_8_hsis,
    #                     labels=["ccsds123", "hcs1d", "hybrid"],
    #                     metrics={"ccsds123": {"CR": 8.88, "RMSE": 11.48, "SAM": 0.64},
    #                              "hcs1d": {"CR": 8.25, "RMSE": 46.08, "SAM": 1.56},
    #                              "hybrid": {"CR": 8.70, "RMSE": 46.95, "SAM": 1.53}})

    cr_22_map = [
        {"run_id": 36, "method": "ccsds123"},
        {"run_id": 456, "method": "hcs1d"},
        {"run_id": 7, "method": "hybrid"},
    ]
    cr_22_hsis = data_processing.load_reconstructed_hsis(has_rec, criteria=cr_22_map)

    # visuals.compare_rgb(cr_22_hsis,
    #                     labels=["ccsds123", "hcs1d", "hybrid"],
    #                     metrics={"ccsds123": {"CR": 13.04, "RMSE": 170.87, "SAM": 8.69},
    #                              "hcs1d": {"CR": 22.00, "RMSE": 78.16, "SAM": 2.76},
    #                              "hybrid": {"CR": 21.99, "RMSE": 58.86, "SAM": 2.13}})
    
    cr_22_hsis.insert(0, original)
    cr_8_hsis.insert(0, original)

    sand_pixel = (57, 177)
    grass_pixel = (122, 80)

    visuals.compare_spectra(cr_22_hsis, 
                            labels=["original", "ccsds123", "hcs1d", "hybrid"], 
                            pixel=grass_pixel,
                            style=visuals.DEFAULT_STYLE)


    plt.show()


if __name__ == "__main__":
    main()
