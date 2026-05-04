import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLOR_MAP = {
    "Original": "#000000",
    "CCSDS123": '#1f77b4',
    "HCS1D": "#d10000",
    "OTHER": "#157c0b"
}

def get_metric_series(csv_path, x_metric, y_metric, label, filters=None):
    df = pd.read_csv(csv_path)

    if filters:
        for k, v in filters.items():
            df = df[df[k] == v]

    df = df.dropna(subset=[x_metric, y_metric])

    return {
        "x": df[x_metric],
        "y": df[y_metric],
        "label": label
    }

def get_averaged_metric_series(csv_path, x_metric, y_metric, label, groupby_cols, filters=None):
    df = pd.read_csv(csv_path)

    # Optional filtering
    if filters:
        for k, v in filters.items():
            df = df[df[k] == v]

    # Drop missing values
    df = df.dropna(subset=[x_metric, y_metric])

    # Group and average
    grouped = (
        df
        .groupby(groupby_cols)
        .agg({
            x_metric: ["mean", "std"],
            y_metric: ["mean", "std"]
        })
    )

    # flatten column names
    grouped.columns = ["_".join(col) for col in grouped.columns]
    grouped = grouped.reset_index()


    return {
        "x": grouped[f"{x_metric}_mean"],
        "y": grouped[f"{y_metric}_mean"],
        "xerr": grouped[f"{x_metric}_std"],
        "yerr": grouped[f"{y_metric}_std"],
        "label": label
    }

def plot_multiple_series(series_list, x_label, y_label, connect_points=False, show_error=True):
    plt.figure(figsize=(8, 4.5))

    legend_handles = []
    for s in series_list:
        color = COLOR_MAP.get(s["label"], "black")
        
        # Determine if we pass error bars or not
        y_err_val = s.get("yerr") if show_error else None

        plt.errorbar(
            s["x"],
            s["y"],
            yerr=y_err_val,
            fmt= '-o' if connect_points else 'o',
            capsize=3 if show_error else 0,
            color=color,
            # If show_error is False, the bars effectively disappear
            elinewidth=1 if show_error else 0,
            markeredgewidth=1 if show_error else 0
        )

        handle = Line2D(
            [0], [0],
            marker='o',
            linestyle='-' if connect_points else 'None',
            color=color,
            label=s["label"]
        )
        legend_handles.append(handle)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{y_label} vs {x_label}")
    plt.legend(handles=legend_handles, loc="upper left")
    plt.grid(alpha=0.3)
    plt.show()
    
def fetch_recent(csv_path):
    """
    Returns most recent log row from csv_path as a dictionary.
    """
    df = pd.read_csv(csv_path)

    # Parse timestamps safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Remove invalid timestamps (optional but safer)
    df = df.dropna(subset=["timestamp"])

    # Get most recent row
    recent_row = df.loc[df["timestamp"].idxmax()]

    # Convert to dictionary
    return recent_row.to_dict()