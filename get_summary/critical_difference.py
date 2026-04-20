"""
Generate Demsar-style critical difference plots for individual metrics.

Metrics:
Set 1 (main language quality): PPL, Dist_2, TTR, VR, CD
Set 2 (discourse/continuity): FDR, CSR, CECPR, DCPR, CCPR, TTCPR, VCPR
"""

import math
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from get_summary.summary_table import (
    create_summary_table,
    get_group_method_exclusions,
    get_metric_groups,
)


with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    RESULTS = config_data["RESULTS"]
    SUMMARY_FOLDER = config_data["SUMMARY_FOLDER"]


def _q_alpha_nemenyi(k: int, alpha: float = 0.05) -> float:
    """
    Return q_alpha used in Nemenyi CD.

    Uses scipy if available; otherwise falls back to a small lookup table
    (values already divided by sqrt(2), as commonly reported for CD formula).
    """
    try:
        from scipy.stats import studentized_range

        return float(studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2.0))
    except Exception:
        fallback = {
            2: 1.960,
            3: 2.344,
            4: 2.569,
            5: 2.728,
            6: 2.850,
            7: 2.949,
            8: 3.031,
            9: 3.102,
            10: 3.164,
        }
        if k in fallback:
            return fallback[k]
        return fallback[10]


def _build_blocks(summary_df: pd.DataFrame, metric_directions: Dict[str, str]) -> pd.DataFrame:
    """
    Build ranking blocks (dataset x metric), with one rank row per block.

    Ranks are computed among methods with non-missing values for that block.
    Methods with missing values are penalised by receiving an equal last rank
    (i.e. max_rank_of_present_methods + 1), shared among all absent methods.
    """
    records = []
    methods = sorted(summary_df["Method"].dropna().astype(str).unique().tolist())

    for dataset_name, dataset_block in summary_df.groupby("Dataset", sort=True):
        for metric, direction in metric_directions.items():
            if metric not in dataset_block.columns:
                continue

            block = dataset_block[["Method", metric]].copy()
            block_present = block.dropna(subset=[metric])
            block_missing = block[block[metric].isna()]

            if block_present["Method"].nunique() < 1:
                continue

            ascending = direction == "down"

            block_present = block_present.copy()
            block_present["Rank"] = block_present[metric].rank(
                ascending=ascending, method="average"
            )

            if not block_missing.empty:
                penalty_rank = block_present["Rank"].max() + 1
                block_missing = block_missing.copy()
                block_missing["Rank"] = penalty_rank

            ranked = pd.concat([block_present, block_missing], ignore_index=True)

            if ranked["Method"].nunique() < 2:
                continue

            row = {"Dataset": dataset_name, "Metric": metric}
            for m in methods:
                row[m] = np.nan
            for _, r in ranked.iterrows():
                row[str(r["Method"])] = float(r["Rank"])

            records.append(row)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def compute_cd_inputs(summary_df: pd.DataFrame, metric_directions: Dict[str, str]) -> Tuple[pd.DataFrame, int, int]:
    """
    Return average ranks and CD inputs (N blocks, k methods with valid ranks).
    """
    rank_blocks = _build_blocks(summary_df, metric_directions)
    if rank_blocks.empty:
        return pd.DataFrame(columns=["Method", "Average_Rank", "Num_Blocks"]), 0, 0

    method_cols = [c for c in rank_blocks.columns if c not in {"Dataset", "Metric"}]
    long_df = rank_blocks.melt(
        id_vars=["Dataset", "Metric"], value_vars=method_cols, var_name="Method", value_name="Rank"
    ).dropna(subset=["Rank"])

    avg_rank_df = (
        long_df.groupby("Method", as_index=False)
        .agg(Average_Rank=("Rank", "mean"), Num_Blocks=("Rank", "size"))
        .sort_values("Average_Rank", ascending=True)
        .reset_index(drop=True)
    )
    avg_rank_df["Average_Rank"] = avg_rank_df["Average_Rank"].round(3)

    n_blocks = rank_blocks.shape[0]
    k_methods = avg_rank_df.shape[0]
    return avg_rank_df, n_blocks, k_methods


def plot_critical_difference(
    avg_rank_df: pd.DataFrame,
    n_blocks: int,
    output_png: str,
    title: str,
    alpha: float = 0.05,
) -> float:
    """Create and save a simple Demsar-style critical difference diagram."""
    if avg_rank_df.empty or n_blocks <= 0 or avg_rank_df.shape[0] < 2:
        raise ValueError("Not enough data to draw CD plot.")

    avg_rank_df = avg_rank_df.sort_values("Average_Rank", ascending=True).reset_index(drop=True)
    methods = avg_rank_df["Method"].tolist()
    methods_display = [_FRAMEWORK_LABELS.get(m, m) for m in methods]
    avg_ranks = avg_rank_df["Average_Rank"].to_numpy(dtype=float)

    k = len(methods)
    q_alpha = _q_alpha_nemenyi(k, alpha=alpha)
    cd = q_alpha * math.sqrt((k * (k + 1)) / (6.0 * n_blocks))

    min_rank = 1.0
    max_rank = max(float(np.nanmax(avg_ranks)) + 0.8, float(k) + 0.2)

    fig, ax = plt.subplots(figsize=(13, 4.8))
    ax.set_title(title)
    ax.set_xlim(min_rank - 0.1, max_rank)
    ax.set_ylim(0, 1)
    ax.get_yaxis().set_visible(False)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    ax.set_xlabel("Average rank (lower is better)")
    ax.set_xticks(np.arange(1, int(math.ceil(max_rank)) + 1, 1))
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    y_base = 0.78
    ax.hlines(y_base, min_rank, max_rank - 0.1, color="black", linewidth=1.2)

    y_start = 0.64
    y_step = 0.06
    y_min = 0.12

    split = int(math.ceil(k / 2))
    left_side = list(range(split))
    right_side = list(range(split, k))

    for local_idx, pos in enumerate(left_side):
        method_name = methods_display[pos]
        x = avg_ranks[pos]
        y = max(y_start - local_idx * y_step, y_min)
        ax.vlines(x, y_base, y, color="black", linewidth=1.0)
        ax.hlines(y, min_rank, x, color="black", linewidth=0.8)
        ax.text(min_rank - 0.02, y, method_name, ha="right", va="center", fontsize=9)

    for local_idx, pos in enumerate(right_side):
        method_name = methods_display[pos]
        x = avg_ranks[pos]
        y = max(y_start - local_idx * y_step, y_min)
        ax.vlines(x, y_base, y, color="black", linewidth=1.0)
        ax.hlines(y, x, max_rank - 0.1, color="black", linewidth=0.8)
        ax.text(max_rank, y, method_name, ha="left", va="center", fontsize=9)

    x0 = min_rank + 0.2
    x1 = x0 + cd
    y_cd = 0.98
    ax.hlines(y_cd, x0, x1, color="black", linewidth=2.0)
    ax.vlines([x0, x1], y_cd - 0.02, y_cd + 0.02, color="black", linewidth=1.6)
    ax.text((x0 + x1) / 2, y_cd + 0.01, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=9)

    y_clique = 0.74
    for i in range(k):
        j = i
        while j + 1 < k and (avg_ranks[j + 1] - avg_ranks[i]) <= cd:
            j += 1
        if j > i:
            ax.hlines(y_clique, avg_ranks[i], avg_ranks[j], color="black", linewidth=3.0)
            y_clique -= 0.035

    footer = f"N blocks={n_blocks}, k methods={k}, alpha={alpha}"
    ax.text(min_rank, 0.03, footer, ha="left", va="bottom", fontsize=8)

    fig.tight_layout()
    base = output_png.removesuffix(".png")
    fig.savefig(f"{base}.png", dpi=240, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)

    return cd


_METRIC_DISPLAY_NAMES: Dict[str, str] = {
    "PPL": "PPL",
    "Bigram_ratio": "Dist$_2$",
    "TTR": "TTR",
    "Verb_ratio": "VR",
    "Conn_all": "CD",
    "FDR": "FDR",
    "CSR": "CSR",
    "CECPR": "CECPR",
    "DCPR": "DCPR",
    "CCPR": "CCPR",
    "TTCPR": "TTCPR",
    "VCPR": "VCPR",
}

_FRAMEWORK_LABELS: Dict[str, str] = {
    "templated_narrative": "Templated Explanation",
    "xaistories_narratives": "XAIstories Narratives",
    "xaistories": "XAIstories",
    "explingo_narratives": "Explingo Narratives",
    "explingo_zero_shot": "Explingo Zero-Shot",
    "talktomodel": "TalkToModel",
    "explingo": "Explingo",
}

# Canonical full order — subsets preserve this ordering
_METRIC_FULL_ORDER = [
    "PPL", "Bigram_ratio", "TTR", "Verb_ratio", "Conn_all",
    "FDR", "CSR", "CECPR", "DCPR", "CCPR", "TTCPR", "VCPR",
]

# Named metric subsets used for the breakdown tables
METRIC_SUBSETS: Dict[str, list] = {
    "all": _METRIC_FULL_ORDER,
    "language_quality": ["PPL", "Bigram_ratio", "TTR", "Verb_ratio", "Conn_all"],
    "discourse": ["FDR", "CSR", "CECPR", "DCPR", "CCPR", "TTCPR", "VCPR"],
}

_SUBSET_CAPTIONS: Dict[str, str] = {
    "all": "all metrics",
    "language_quality": "standard NLP metrics (PPL, Dist$_2$, TTR, VR, CD)",
    "discourse": "narrativity metrics (FDR, CSR, CECPR, DCPR, CCPR, TTCPR, VCPR)",
}


def generate_summary_ranks_table(
    summary_df: pd.DataFrame,
    groups: Dict,
    output_tables: str,
    metric_subset: str = "all",
) -> None:
    """
    Generate a table of per-metric average ranks (averaged over datasets).

    Parameters
    ----------
    metric_subset : one of ``"all"``, ``"language_quality"``, ``"discourse"``.
        Controls which metrics are included in the per-column ranks and the
        Overall average.  Rankings are always recomputed from scratch within
        the chosen subset so that the Overall column is consistent.

    Missing values are penalised: each absent method receives an equal last rank
    within that (dataset, metric) block before averaging.
    """
    if metric_subset not in METRIC_SUBSETS:
        raise ValueError(
            f"Unknown metric_subset {metric_subset!r}. Choose from {list(METRIC_SUBSETS)}"
        )

    metric_display_order = [m for m in METRIC_SUBSETS[metric_subset] if m in _METRIC_FULL_ORDER]

    all_metrics: Dict[str, str] = {}
    for metric_directions in groups.values():
        for metric, direction in metric_directions.items():
            all_metrics[metric] = direction

    results: Dict[str, Dict[str, float]] = {}
    for metric in metric_display_order:
        if metric not in all_metrics or metric not in summary_df.columns:
            continue

        avg_rank_df, _, _ = compute_cd_inputs(summary_df, {metric: all_metrics[metric]})

        for _, row in avg_rank_df.iterrows():
            method = row["Method"]
            results.setdefault(method, {})[metric] = row["Average_Rank"]

    all_methods = sorted(results.keys())

    overall_ranks: Dict[str, float] = {}
    for method in all_methods:
        vals = [
            results[method][m]
            for m in metric_display_order
            if m in results[method] and not pd.isna(results[method][m])
        ]
        overall_ranks[method] = round(sum(vals) / len(vals), 3) if vals else float("nan")

    all_methods_sorted = sorted(all_methods, key=lambda m: overall_ranks.get(m, float("inf")))

    # ── CSV ──────────────────────────────────────────────────────────────────
    rows = []
    for method in all_methods_sorted:
        method_display = _FRAMEWORK_LABELS.get(method, method)
        row: Dict = {"Method": method_display}
        for metric in metric_display_order:
            row[_METRIC_DISPLAY_NAMES[metric]] = results.get(method, {}).get(metric, None)
        row["Overall"] = overall_ranks.get(method, None)
        rows.append(row)

    df_out = pd.DataFrame(rows)
    csv_path = os.path.join(output_tables, f"summary_average_ranks_{metric_subset}.csv")
    df_out.to_csv(csv_path, index=False)
    print(f"\nSummary table [{metric_subset}] (CSV): {csv_path}")

    # ── LaTeX ────────────────────────────────────────────────────────────────
    display_cols = [_METRIC_DISPLAY_NAMES[m] for m in metric_display_order]

    def _col_min(metric: str) -> float:
        vals = [results.get(m, {}).get(metric) for m in all_methods_sorted]
        vals = [v for v in vals if v is not None and not pd.isna(v)]
        return min(vals) if vals else float("nan")

    col_mins = {m: _col_min(m) for m in metric_display_order}
    overall_min = min(
        (v for v in overall_ranks.values() if not pd.isna(v)), default=float("nan")
    )

    caption_detail = _SUBSET_CAPTIONS[metric_subset]
    tex_lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        (
            f"\\caption{{Average rank per metric across datasets ({caption_detail}; lower is better). "
            r"Missing values are penalised with equal last rank within each block. "
            r"Overall = mean of per-metric average ranks. Best value per column in \textbf{bold}.}"
        ),
        f"\\label{{tab:average_ranks_{metric_subset}}}",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{0.85}",
        r"\resizebox{\textwidth}{!}{",
    ]

    col_spec = "l" + "r" * (len(metric_display_order) + 1)
    tex_lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    tex_lines.append(r"    \toprule")
    tex_lines.append(
        "    " + " & ".join(["Method"] + display_cols + [r"\textbf{Overall}"]) + r" \\"
    )
    tex_lines.append(r"    \midrule")

    for method in all_methods_sorted:
        method_display = _FRAMEWORK_LABELS.get(method, method)
        cells = [method_display]
        for metric in metric_display_order:
            val = results.get(method, {}).get(metric)
            if val is None or pd.isna(val):
                cells.append("--")
            else:
                fmt = f"{val:.3f}"
                cells.append(f"\\textbf{{{fmt}}}" if val == col_mins[metric] else fmt)

        ov = overall_ranks.get(method)
        if ov is None or pd.isna(ov):
            cells.append("--")
        else:
            fmt = f"{ov:.3f}"
            cells.append(f"\\textbf{{{fmt}}}" if ov == overall_min else fmt)

        tex_lines.append("    " + " & ".join(cells) + r" \\")

    tex_lines += [r"    \bottomrule", r"    \end{tabular}", r"}", r"\end{table}"]

    tex_path = os.path.join(output_tables, f"summary_average_ranks_{metric_subset}.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Summary table [{metric_subset}] (LaTeX): {tex_path}")


def _collect_rank_matrix(
    summary_df: pd.DataFrame,
    groups: Dict,
    metric_display_order: list,
) -> tuple:
    """
    Return (rank_matrix, methods_sorted, display_col_names, overall) for a given metric order.

    rank_matrix shape: (n_methods, n_metrics).  Overall avg rank is used for row sorting.
    """
    all_metrics: Dict[str, str] = {}
    for metric_directions in groups.values():
        for metric, direction in metric_directions.items():
            all_metrics[metric] = direction

    results: Dict[str, Dict[str, float]] = {}
    for metric in metric_display_order:
        if metric not in all_metrics or metric not in summary_df.columns:
            continue
        avg_rank_df, _, _ = compute_cd_inputs(summary_df, {metric: all_metrics[metric]})
        for _, row in avg_rank_df.iterrows():
            results.setdefault(row["Method"], {})[metric] = row["Average_Rank"]

    all_methods = sorted(results.keys())
    overall = {
        m: np.nanmean([results[m].get(met, np.nan) for met in metric_display_order])
        for m in all_methods
    }
    methods_sorted = sorted(all_methods, key=lambda m: overall[m])

    matrix = np.array(
        [[results.get(m, {}).get(met, np.nan) for met in metric_display_order]
         for m in methods_sorted],
        dtype=float,
    )
    col_names = [_METRIC_DISPLAY_NAMES[m] for m in metric_display_order]
    return matrix, methods_sorted, col_names, overall


def plot_rank_heatmap(
    summary_df: pd.DataFrame,
    groups: Dict,
    output_png: str,
    metric_subset: str = "all",
) -> None:
    """
    Save a rank heatmap (methods × metrics) for the given metric subset.

    Colour scale runs green (rank 1) → red (worst rank); colorbar has rank 1
    at the top.  Saved as PNG and PDF.
    """
    from matplotlib.colors import LinearSegmentedColormap

    metric_display_order = METRIC_SUBSETS[metric_subset]
    matrix, methods, col_names, overall = _collect_rank_matrix(
        summary_df, groups, metric_display_order
    )

    n_methods, n_metrics = matrix.shape
    overall_col = np.array([overall[m] for m in methods]).reshape(-1, 1)

    full_matrix = np.hstack([matrix, np.full((n_methods, 1), np.nan), overall_col])
    full_col_names = col_names + [""] + ["Overall"]

    fig_w = max(9, len(full_col_names) * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, max(3.2, n_methods * 0.62)))

    vmin, vmax = 1.0, float(n_methods)
    green_red_cmap = LinearSegmentedColormap.from_list(
        "rank_green_red", ["#1a7f2e", "#f7f7b6", "#b2182b"]
    )

    col_widths = [1.0] * n_metrics + [0.5] + [1.0]
    x_edges = np.concatenate(([-0.5], -0.5 + np.cumsum(col_widths)))
    y_edges = np.arange(-0.5, n_methods + 0.5, 1.0)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2

    X, Y = np.meshgrid(x_edges, y_edges)
    im = ax.pcolormesh(X, Y, full_matrix, cmap=green_red_cmap, vmin=vmin, vmax=vmax, shading="flat")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(n_methods - 0.5, -0.5)

    for r in range(n_methods):
        for c in range(len(full_col_names)):
            val = full_matrix[r, c]
            if np.isnan(val) or full_col_names[c] == "":
                continue
            norm_val = (val - vmin) / max(vmax - vmin, 1)
            txt_color = "white" if (norm_val < 0.25 or norm_val > 0.75) else "#1a1a1a"
            weight = "bold" if full_col_names[c] == "Overall" else "normal"
            ax.text(x_centers[c], r, f"{val:.2f}", ha="center", va="center",
                    fontsize=8.5, color=txt_color, fontweight=weight)

    ax.axvline(x_edges[-2], color="white", linewidth=2)
    if metric_subset == "all":
        ax.axvline(4.5, color="black", linewidth=2)

    ax.set_xticks(x_centers)
    ax.set_xticklabels(full_col_names, fontsize=10, rotation=30, ha="right")
    ax.set_yticks(range(n_methods))
    methods_display = [_FRAMEWORK_LABELS.get(m, m) for m in methods]
    ax.set_yticklabels(methods_display, fontsize=10)
    ax.tick_params(length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.invert_yaxis()
    cbar.set_ticks([vmin, (vmin + vmax) / 2, vmax])
    cbar.set_ticklabels([f"1 (best)", f"{(vmin + vmax) / 2:.1f}", f"{int(vmax)} (worst)"])
    cbar.set_label("Average Rank", fontsize=10)
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    base = output_png.removesuffix(".png")
    fig.savefig(f"{base}.png", dpi=240, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Rank heatmap [{metric_subset}]: {base}.png / .pdf")


def plot_rank_overview(
    summary_df: pd.DataFrame,
    groups: Dict,
    output_png: str,
) -> None:
    """
    Save a grouped bar chart comparing Overall avg rank per method across the
    two metric subsets (standard NLP vs narrativity) and the grand overall.

    Methods are sorted by their grand-overall rank on the x-axis.
    Saved as PNG and PDF.
    """
    subset_keys   = ["language_quality",     "discourse",           "all"]
    subset_labels = ["Standard NLP Metrics", "Narrativity Metrics", "All Metrics Combined"]
    colors        = ["#E69F00",              "#0072B2",             "#CC79A7"]

    data: Dict[str, Dict[str, float]] = {}
    for subset in subset_keys:
        _, methods, _, overall = _collect_rank_matrix(
            summary_df, groups, METRIC_SUBSETS[subset]
        )
        data[subset] = overall

    all_methods = sorted(data["all"].keys(), key=lambda m: data["all"][m])
    n = len(all_methods)
    x = np.arange(n)
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), 4.5))

    for i, (subset, label, color) in enumerate(zip(subset_keys, subset_labels, colors)):
        vals = [data[subset].get(m, np.nan) for m in all_methods]
        bars = ax.bar(x + (i - 1) * bar_w, vals, bar_w, label=label,
                      color=color, alpha=0.88, zorder=3)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([_FRAMEWORK_LABELS.get(m, m) for m in all_methods],
                       rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("Average Rank (↓ Better)", fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, framealpha=0.6)

    fig.tight_layout()
    base = output_png.removesuffix(".png")
    fig.savefig(f"{base}.png", dpi=240, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"Rank overview bar chart: {base}.png / .pdf")


def generate_nemenyi_posthoc_table(
    summary_df: pd.DataFrame,
    groups: Dict,
    output_tables: str,
    alpha: float = 0.05,
) -> None:
    """
    Generate Nemenyi post-hoc test results table for all metrics.

    For each metric only methods present in ALL datasets are included so that
    per-dataset rank arrays have the same length (required by Friedman's test).
    Methods missing from any dataset are excluded from significance testing for
    that metric and marked with a dagger in the output table.
    """
    from scipy.stats import friedmanchisquare

    all_metrics: Dict[str, str] = {}
    for metric_directions in groups.values():
        for metric, direction in metric_directions.items():
            all_metrics[metric] = direction

    metric_display_order = [
        "PPL", "Bigram_ratio", "TTR", "Verb_ratio", "Conn_all",
        "FDR", "CSR", "CECPR", "DCPR", "CCPR", "TTCPR", "VCPR",
    ]

    datasets = sorted(summary_df["Dataset"].unique())

    # ── Build per-metric rank matrices ────────────────────────────────────────
    results: Dict[str, dict] = {}

    for metric in metric_display_order:
        if metric not in all_metrics or metric not in summary_df.columns:
            continue

        direction = all_metrics[metric]
        ascending = direction == "down"

        # Use ALL known methods, applying the same penalty-rank logic as _build_blocks:
        # methods present in a dataset are ranked normally; methods absent get an equal
        # last rank (max_present_rank + 1).  This keeps the matrix homogeneous without
        # excluding any method from significance testing.
        all_known_methods = sorted(
            summary_df["Method"].dropna().astype(str).unique().tolist()
        )
        common_methods = all_known_methods  # kept for compatibility with results dict

        if len(common_methods) < 2:
            print(f"Skipping {metric} for Friedman/Nemenyi: fewer than 2 methods.")
            continue

        rank_matrix = []
        for dataset in datasets:
            ds = summary_df[summary_df["Dataset"] == dataset].copy()

            # Left-join from ALL known methods so every method appears in the block,
            # with NaN for those absent from this dataset or missing the metric value.
            # This guarantees a single penalty level — consistent with _build_blocks.
            full_block = pd.DataFrame({"Method": common_methods})
            ds_metric = ds[["Method", metric]].copy()
            ds_metric["Method"] = ds_metric["Method"].astype(str)
            full_block = full_block.merge(ds_metric, on="Method", how="left")

            block_present = full_block.dropna(subset=[metric]).copy()
            block_missing = full_block[full_block[metric].isna()].copy()

            if block_present.empty:
                continue

            block_present["Rank"] = block_present[metric].rank(
                ascending=ascending, method="average"
            )

            if not block_missing.empty:
                penalty = block_present["Rank"].max() + 1
                block_missing["Rank"] = penalty

            ranked = (
                pd.concat([block_present, block_missing], ignore_index=True)
                .set_index("Method")
                .loc[common_methods]
            )
            rank_matrix.append(ranked["Rank"].values.astype(float))

        if not rank_matrix:
            continue

        # Every row has len(common_methods) entries → homogeneous shape guaranteed
        rank_matrix = np.array(rank_matrix, dtype=float)  # (n_datasets, n_methods)

        try:
            friedman_stat, friedman_p = friedmanchisquare(*rank_matrix.T)
        except Exception as e:
            print(f"Warning: Friedman test failed for {metric}: {e}")
            continue

        avg_ranks = rank_matrix.mean(axis=0)
        n_ds, k = rank_matrix.shape
        q_alpha = _q_alpha_nemenyi(k, alpha=alpha)
        cd = q_alpha * math.sqrt((k * (k + 1)) / (6.0 * n_ds))

        results[metric] = {
            "friedman_stat": friedman_stat,
            "friedman_p": friedman_p,
            "avg_ranks": avg_ranks,
            "cd": cd,
            "n_datasets": n_ds,
            "k_methods": k,
            "common_methods": common_methods,
        }

    # ── Pairwise significance table ───────────────────────────────────────────
    from scipy.stats import studentized_range as _srange

    def _nemenyi_pvalue(rank_diff: float, k: int, n_ds: int) -> float:
        """Two-sided Nemenyi p-value from the studentized range distribution."""
        se = math.sqrt(k * (k + 1) / (6.0 * n_ds))
        if se == 0:
            return 1.0
        q_stat = rank_diff * math.sqrt(2) / se
        return float(1.0 - _srange.cdf(q_stat, k, np.inf))

    # Pairs are generated in the user-specified display order.
    # The order controls both row order and which method appears first in each pair label.
    _NEMENYI_METHOD_ORDER = [
        "talktomodel",
        "explingo",
        "templated_narrative",
        "xaistories",
        "xaistories_narratives",
        "explingo_zero_shot",
        "explingo_narratives",
    ]
    all_known = summary_df["Method"].dropna().astype(str).unique().tolist()
    # Start from the specified order, append any unlisted methods at the end
    all_methods_sorted = _NEMENYI_METHOD_ORDER + [
        m for m in sorted(all_known) if m not in _NEMENYI_METHOD_ORDER
    ]
    # Keep only methods actually present in the data
    all_methods_sorted = [m for m in all_methods_sorted if m in all_known]

    pairs = [
        (all_methods_sorted[i], all_methods_sorted[j])
        for i in range(len(all_methods_sorted))
        for j in range(i + 1, len(all_methods_sorted))
    ]

    valid_metrics = [m for m in metric_display_order if m in results]

    col_spec = "l" + "c" * len(valid_metrics)
    tex_lines = [
        r"% Requires \usepackage[table]{xcolor} in your preamble.",
        r"\begin{table}[!htb]",
        r"\centering",
        (
            r"\caption{Nemenyi post-hoc pairwise $p$-values. "
            r"Significance: $^{***}\,p<0.001$, $^{**}\,p<0.01$, $^{*}\,p<0.05$. "
            r"Methods with missing results on a dataset receive an equal last rank penalty "
            r"(consistent with the ranking tables).}"
        ),
        r"\label{tab:nemenyi_posthoc}",
        r"\setlength{\tabcolsep}{3pt}",
        r"\renewcommand{\arraystretch}{0.9}",
        r"\resizebox{\textwidth}{!}{",
        f"    \\begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    header_cols = ["Framework pair"] + [_METRIC_DISPLAY_NAMES.get(m, m) for m in valid_metrics]
    tex_lines.append("    " + " & ".join(header_cols) + r" \\")
    tex_lines.append(r"    \midrule")

    # Darker → more significant
    COLOR_001 = "4CAF50"  # p < 0.001 (medium-dark green)
    COLOR_01 = "90EE90"   # p < 0.01  (light green)
    COLOR_05 = "DFF5DC"   # p < 0.05  (very light green)

    for method1, method2 in pairs:
        m1_display = _FRAMEWORK_LABELS.get(method1, method1)
        m2_display = _FRAMEWORK_LABELS.get(method2, method2)
        cells = [f"{m1_display} vs {m2_display}"]

        for metric in valid_metrics:
            r = results[metric]

            if r["friedman_p"] >= alpha:
                cells.append("--")
                continue

            common = r["common_methods"]
            idx1 = common.index(method1)
            idx2 = common.index(method2)
            rank_diff = abs(r["avg_ranks"][idx1] - r["avg_ranks"][idx2])
            p_val = _nemenyi_pvalue(rank_diff, r["k_methods"], r["n_datasets"])

            if p_val < 0.001:
                p_str = r"$<$0.001"
                stars = r"$^{***}$"
                color = COLOR_001
            elif p_val < 0.01:
                p_str = f"{p_val:.3f}"
                stars = r"$^{**}$"
                color = COLOR_01
            elif p_val < 0.05:
                p_str = f"{p_val:.3f}"
                stars = r"$^{*}$"
                color = COLOR_05
            else:
                p_str = f"{p_val:.3f}"
                stars = ""
                color = None

            cell_text = f"{p_str}{stars}"
            if color is not None:
                cells.append(f"\\cellcolor[HTML]{{{color}}}{cell_text}")
            else:
                cells.append(cell_text)

        tex_lines.append("    " + " & ".join(cells) + r" \\")

    tex_lines += [
        r"    \bottomrule",
        r"    \end{tabular}",
        r"}",
        r"\end{table}",
    ]

    tex_path = os.path.join(output_tables, "nemenyi_posthoc_pairwise.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"Nemenyi post-hoc table (LaTeX): {tex_path}")

    # ── Friedman summary table ────────────────────────────────────────────────
    friedman_lines = [
        r"\begin{table}[!htb]",
        r"\centering",
        (
            r"\caption{Friedman test summary. "
            r"For each metric, tests the null hypothesis that all frameworks have equal "
            r"average ranks. Methods with missing results receive an equal last rank penalty. "
            r"$p < 0.05$ indicates significant differences exist.}"
        ),
        r"\label{tab:friedman_summary}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\begin{tabular}{lrrrr}",
        r"    \toprule",
        r"    Metric & $\chi^2$ & $df$ & $p$-value & Significant \\",
        r"    \midrule",
    ]

    for metric in metric_display_order:
        if metric not in results:
            continue
        r = results[metric]
        sig_marker = r"\checkmark" if r["friedman_p"] < alpha else "—"
        metric_display = _METRIC_DISPLAY_NAMES.get(metric, metric)
        friedman_lines.append(
            f"    {metric_display} & {r['friedman_stat']:.2f} & "
            f"{r['k_methods'] - 1} & {r['friedman_p']:.3f} & "
            f"{sig_marker} \\\\"
        )

    friedman_lines += [
        r"    \bottomrule",
        r"    \end{tabular}",
        r"\end{table}",
    ]

    friedman_path = os.path.join(output_tables, "friedman_test_summary.tex")
    with open(friedman_path, "w") as f:
        f.write("\n".join(friedman_lines) + "\n")
    print(f"Friedman test summary (LaTeX): {friedman_path}")


def main() -> None:
    summary_df = create_summary_table(results_folder=RESULTS)
    output_tables = os.path.join(SUMMARY_FOLDER, "tables")
    output_charts = os.path.join(SUMMARY_FOLDER, "charts", "critical_difference")
    os.makedirs(output_tables, exist_ok=True)
    os.makedirs(output_charts, exist_ok=True)

    groups = get_metric_groups()
    method_exclusions = get_group_method_exclusions()

    metric_display_names = {
        "PPL": "PPL",
        "Bigram_ratio": "Dist₂",
        "TTR": "TTR",
        "Verb_ratio": "VR",
        "Conn_all": "CD",
        "FDR": "FDR",
        "CSR": "CSR",
        "CECPR": "CECPR",
        "DCPR": "DCPR",
        "CCPR": "CCPR",
        "TTCPR": "TTCPR",
        "VCPR": "VCPR",
    }

    # Collect all metrics and their directions from groups
    all_metrics: Dict[str, Tuple[str, str]] = {}
    for group_name, metric_directions in groups.items():
        for metric, direction in metric_directions.items():
            all_metrics[metric] = (direction, group_name)

    # CD plots per individual metric
    for metric, (direction, group_name) in all_metrics.items():
        if metric not in summary_df.columns:
            print(f"Skipping {metric}: not in summary table.")
            continue

        metric_directions = {metric: direction}
        avg_rank_df, n_blocks, k_methods = compute_cd_inputs(summary_df, metric_directions)
        if avg_rank_df.empty or n_blocks == 0 or k_methods < 2:
            print(f"Skipping {metric}: insufficient data for CD plot.")
            continue

        display_name = metric_display_names.get(metric, metric)
        metric_key = metric.lower()

        csv_path = os.path.join(output_tables, f"summary_average_ranks_{metric_key}.csv")
        tex_path = os.path.join(output_tables, f"summary_average_ranks_{metric_key}.tex")
        avg_rank_df.to_csv(csv_path, index=False)
        avg_rank_df.to_latex(tex_path, index=False, float_format="%.3f")

        png_path = os.path.join(output_charts, f"critical_difference_{metric_key}.png")
        cd = plot_critical_difference(
            avg_rank_df=avg_rank_df,
            n_blocks=n_blocks,
            output_png=png_path,
            title=f"Critical Difference Diagram ({display_name})",
            alpha=0.05,
        )

        print(f"{metric}: saved ranks to '{csv_path}' and '{tex_path}'")
        print(f"{metric}: saved CD plot to '{png_path}' + .pdf (CD={cd:.3f}, N={n_blocks}, k={k_methods})")

    # Summary average-rank tables (approach B): all metrics + two breakdowns
    print("\n" + "=" * 80)
    print("Generating summary tables of average ranks...")
    print("=" * 80)
    for subset in ("all", "language_quality", "discourse"):
        generate_summary_ranks_table(summary_df, groups, output_tables, metric_subset=subset)

    # Nemenyi post-hoc test tables
    print("\n" + "=" * 80)
    print("Generating Nemenyi post-hoc test results...")
    print("=" * 80)
    generate_nemenyi_posthoc_table(summary_df, groups, output_tables, alpha=0.05)

    # Rank visualisations
    output_rank_charts = os.path.join(SUMMARY_FOLDER, "charts", "rank_profiles")
    os.makedirs(output_rank_charts, exist_ok=True)

    print("\n" + "=" * 80)
    print("Generating rank visualisations...")
    print("=" * 80)

    for subset in ("all", "language_quality", "discourse"):
        plot_rank_heatmap(
            summary_df, groups,
            output_png=os.path.join(output_rank_charts, f"rank_heatmap_{subset}.png"),
            metric_subset=subset,
        )

    plot_rank_overview(
        summary_df, groups,
        output_png=os.path.join(output_rank_charts, "rank_overview_by_group.png"),
    )


if __name__ == "__main__":
    main()