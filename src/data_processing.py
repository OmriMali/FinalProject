import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


COLOR_MAP = {
    "Original": "#000000",
    "CCSDS123": "#cc6600",
    "HCS1D": "#7f00ff",
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
    
    plt.grid(alpha=0.8, color="white")
    plt.gcf().set_facecolor("#808080")
    plt.gca().set_facecolor("#808080")
    plt.gca().tick_params(colors="white")
    plt.gca().xaxis.label.set_color("white")
    plt.gca().yaxis.label.set_color("white")
    plt.gca().title.set_color("white")
    leg = plt.legend(handles=legend_handles, loc="upper left", facecolor="#414141", edgecolor="white")
    for text in leg.get_texts():
        text.set_color("white")
    for spines in plt.gca().spines.values():
        spines.set_color("white")
    plt.show()
    
def fetch_recent(csv_path, n=1):
    """
    Returns the n most recent log rows from csv_path as a list of dictionaries.
    """
    df = pd.read_csv(csv_path)

    # Parse timestamps safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    # Sort by timestamp (newest first) and take top n
    recent_rows = df.sort_values("timestamp", ascending=False).head(n)

    # Convert to list of dictionaries
    return recent_rows.to_dict(orient='records')