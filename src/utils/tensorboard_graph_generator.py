"""
Generate matplotlib graphs from TensorBoard event files produced during DRL agent training.

Event files live under src/agents/<AGENT_NAME>/tb_logs/<run_dir>/, where each
<run_dir> is named "<run_idx>_config_<config_ident>_<date>_<n>". Configure
AGENT_NAME, CONFIG_IDENTS and METRIC_NAMES below and run this script directly.

For every metric, all runs matching the given config_idents are combined onto a
single plot (one line per run). Each figure is saved as a PNG under
src/utils/graphs/ and, once all metrics have been processed, every figure is
also opened on screen.
"""

import os
import re

import matplotlib.pyplot as plt
from tbparse import SummaryReader

# ----------------------------------------------------------------------------
# Configuration - edit these before running the script
# ----------------------------------------------------------------------------
AGENT_NAME = "PPO_portfolio_allocator_weights"
CONFIG_IDENTS = [
    "00283_config_10019_26_10_02_1",
    "00284_config_10020_26_10_03_1",
    "00285_config_10021_26_10_03_1",
    "00286_config_10022_26_10_03_1"
]
METRIC_NAMES = [
    "validation/excess_return_over_spy_abs_mean"
]
TRAIN_SMOOTHING_WINDOW = 25
# ----------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
TB_LOGS_DIR = os.path.join(REPO_ROOT, "src", "agents", AGENT_NAME, "tb_logs")
GRAPHS_DIR = os.path.join(SCRIPT_DIR, "graphs")

# Run dirs look like "00046_config_10002_26_02_14_1" - capture the ident token.
_DIR_CONFIG_RE = re.compile(r"_config_(?P<ident>[^_]+)_")


def _extract_config_ident(dir_name: str) -> str | None:
    match = _DIR_CONFIG_RE.search(dir_name)
    return match.group("ident") if match else None


def load_scalars(tb_logs_dir: str, config_idents: list[str]):
    """Read scalars for requested config identifiers or complete run names."""
    reader = SummaryReader(tb_logs_dir, extra_columns={"dir_name"})
    df = reader.scalars
    requested = set(config_idents)
    mask = df["dir_name"].apply(
        lambda d: d in requested or _extract_config_ident(d) in requested
    )
    return df[mask]


def plot_metric(df, metric: str, graphs_dir: str):
    """Plot matching runs for a metric and save the figure as a PNG."""
    metric_df = df[df["tag"] == metric]
    if metric_df.empty:
        print(f"[warn] no data found for metric '{metric}', skipping.")
        return None

    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=150)
    colors = plt.get_cmap("tab10").colors
    smooth_train = metric.startswith("train/")

    for run_number, (dir_name, run_df) in enumerate(metric_df.groupby("dir_name")):
        run_df = run_df.sort_values("step")
        ident = _extract_config_ident(dir_name)
        label = f"config_{ident}"
        color = colors[run_number % len(colors)]

        if smooth_train:
            ax.plot(
                run_df["step"],
                run_df["value"],
                color=color,
                alpha=0.3,
                linewidth=0.8,
                label="_nolegend_",
            )
            smoothed_values = run_df["value"].rolling(
                window=TRAIN_SMOOTHING_WINDOW,
                min_periods=1,
                center=True,
            ).mean()
            ax.plot(
                run_df["step"],
                smoothed_values,
                color=color,
                linewidth=2,
                label=f"{label} (rolling mean)",
            )
        else:
            ax.plot(run_df["step"], run_df["value"], color=color, label=label)

    ax.set_title(metric)
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.minorticks_on()
    ax.grid(True, which="major", axis="both", linestyle="-", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", axis="both", linestyle=":", linewidth=0.5, alpha=0.25)
    ax.legend(fontsize="small")
    fig.tight_layout()

    filename = metric.replace("/", "_") + ".png"
    out_path = os.path.join(graphs_dir, filename)
    fig.savefig(out_path)
    print(f"Saved graph for '{metric}' -> {out_path}")
    return fig


def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    if not os.path.isdir(TB_LOGS_DIR):
        raise FileNotFoundError(f"tb_logs directory not found: {TB_LOGS_DIR}")

    df = load_scalars(TB_LOGS_DIR, CONFIG_IDENTS)
    if df.empty:
        raise ValueError(
            f"No tensorboard runs found for agent '{AGENT_NAME}' and config_idents {CONFIG_IDENTS}"
        )

    figures = [fig for metric in METRIC_NAMES if (fig := plot_metric(df, metric, GRAPHS_DIR)) is not None]

    if figures:
        plt.show()
    else:
        print("No graphs were generated.")


if __name__ == "__main__":
    main()
