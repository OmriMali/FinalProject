
DARK_STYLE = {
    "figure_facecolor": "#808080",
    "facecolor": "#808080",
    "text_color": "white",
    "grid": True,
    "grid_color": "white",
    "spine_color": "white",
    "legend_facecolor": "#404040",
    "legend_edgecolor": "white",
    "legend_text_color": "white",
    "colors": {
        "hcs1d": "#7F00FF",
        "ccsds123": "#CC6600",
        "hcs3d": "#00AA55",
        "original": "#000000"
    },
}


def apply_axis_style(ax, style: dict | None = None) -> None:
    if style is None:
        return

    if "facecolor" in style:
        ax.set_facecolor(style["facecolor"])

    if style.get("grid", False):
        ax.grid(True, color=style.get("grid_color", None), alpha=0.7)

    text_color = style.get("text_color")
    if text_color is not None:
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)
        ax.title.set_color(text_color)
        ax.tick_params(axis="both", colors=text_color, which="both")
        
        for tick in ax.get_xticklabels():
            tick.set_color(text_color)
        
        for tick in ax.get_yticklabels():
            tick.set_color(text_color)

        ax.yaxis.get_offset_text().set_color(text_color)
        ax.xaxis.get_offset_text().set_color(text_color)

    spine_color = style.get("spine_color")
    if spine_color is not None:
        for spine in ax.spines.values():
            spine.set_color(spine_color)


def apply_figure_style(fig, style: dict | None = None) -> None:
    if style is None:
        return

    if "figure_facecolor" in style:
        fig.set_facecolor(style["figure_facecolor"])


def apply_legend_style(ax, style: dict | None = None) -> None:
    if style is None:
        return

    legend = ax.get_legend()
    if legend is None:
        return

    if "legend_facecolor" in style:
        legend.get_frame().set_facecolor(style["legend_facecolor"])

    if "legend_edgecolor" in style:
        legend.get_frame().set_edgecolor(style["legend_edgecolor"])

    text_color = style.get("legend_text_color", style.get("text_color"))
    if text_color is not None:
        for text in legend.get_texts():
            text.set_color(text_color)