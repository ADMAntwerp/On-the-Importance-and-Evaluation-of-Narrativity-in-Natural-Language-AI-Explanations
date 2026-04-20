"""
Generate bar charts of PPL values w.r.t. PPL_ori.
Loads data from the same JSON results structure as the summary script.
"""

import json
import os
import textwrap

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import yaml


# ── Config ────────────────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    RESULTS = config_data["RESULTS"]
    SUMMARY_FOLDER = config_data["SUMMARY_FOLDER"]


# ── Global plotting style ─────────────────────────────────────────────────────
# Toggle to "log" if your PPL ranges vary a lot across datasets/methods.
PPL_YSCALE = "linear"  # "log" or "linear"
ADD_VALUE_LABELS = False          # bar labels can clutter small subplots
INCLUDE_ERROR_BARS = True         # uses *_std or *_sem if available
SAVE_PDF_TOO = True              # also write vector PDFs (nice for papers)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 16,
    "axes.titlesize": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
})


# ── Load data (mirrors load_ppl_change_ratios_stats from summary script) ──────
def load_ppl_change_ratios_stats(results_folder):
    data = {}
    methods = sorted(
        d for d in os.listdir(results_folder)
        if os.path.isdir(os.path.join(results_folder, d))
    )
    for method in methods:
        method_path = os.path.join(results_folder, method)
        datasets = sorted(
            d for d in os.listdir(method_path)
            if os.path.isdir(os.path.join(method_path, d))
        )
        for dataset in datasets:
            json_path = os.path.join(method_path, dataset, "ppl_change_ratios_stats.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data[f"{method}/{dataset}"] = json.load(f)
    return data


# ── Method / label config ─────────────────────────────────────────────────────
METHOD_ORDER = [
    "templated_narrative",
    "explingo_zero_shot",
    "explingo_narratives",
    "xaistories",
    "xaistories_narratives",
]

SHORT = {
    "templated_narrative":   "Templated\nExpl.",
    "explingo_zero_shot":    "Explingo\n(Z-S)",
    "xaistories":            "XAIstories",
    "explingo_narratives":   "Explingo\n(Narr.)",
    "xaistories_narratives": "XAIstories\n(Narr.)",
}

PPL_SERIES = [
    ("ori",      "Original order",               "#009E73"),
    ("shuffled", "Shuffled order",               "#E69F00"),
    ("reversed", "Reversed order",               "#0072B2"),
    ("loo",      "Leave-one-out sentences",      "#CC79A7"),
]

# Relative increase: compare each transform to ori (3 bars)
RATIO_SERIES = [
    ("shuffled", "Shuffled vs. original",        "#E69F00"),
    ("reversed", "Reversed vs. original",        "#0072B2"),
    ("loo",      "LOO vs. original",             "#CC79A7"),
]


def build_rows(change_ratios_data):
    """
    Convert raw JSON to row dicts:
      {
        dataset, method,
        means: {ori, shuffled, reversed, loo},
        errs:  {ori, shuffled, reversed, loo}  (std/sem if available)
      }
    """
    rows = []
    for key, values in change_ratios_data.items():
        method, dataset = key.split("/")
        ppl = values.get("ppl_values", {})

        means = {}
        errs = {}
        for k, _, _ in PPL_SERIES:
            means[k] = float(ppl.get(f"{k}_mean", 0.0) or 0.0)

            # Prefer std; fall back to sem; else 0.
            std = ppl.get(f"{k}_std", None)
            sem = ppl.get(f"{k}_sem", None)
            if INCLUDE_ERROR_BARS and (std is not None or sem is not None):
                errs[k] = float(std if std is not None else sem)
            else:
                errs[k] = 0.0

        rows.append({
            "dataset": dataset,
            "method": method,
            "means": means,
            "errs": errs,
        })

    rows.sort(key=lambda r: (r["dataset"], r["method"]))
    return rows


# ── Plotting helpers ──────────────────────────────────────────────────────────
def _bar_offsets(n, group_width=0.78):
    # n bars centered in a category, spanning group_width of the category.
    if n <= 1:
        return np.array([0.0]), group_width
    step = group_width / n
    offsets = (np.arange(n) - (n - 1) / 2.0) * step
    return offsets, step * 0.92  # bar width slightly smaller than step


def _legend_patches(labels_colors):
    return [mpatches.Patch(color=c, label=l) for l, c in labels_colors]


def _wrap_title(s, width=24):
    s = s.replace("_", " ").title()
    return "\n".join(textwrap.wrap(s, width=width))


def _setup_ax(ax):
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.tick_params(axis="x", length=0)
    # keep left/bottom spines visible for readability
    ax.spines["left"].set_alpha(0.8)
    ax.spines["bottom"].set_alpha(0.8)


def _apply_ppl_scale(ax):
    if PPL_YSCALE == "log":
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(mticker.LogFormatter())
    else:
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain")


def _save(fig, path_base):
    png_path = f"{path_base}.png"
    fig.savefig(png_path, bbox_inches="tight")
    print(f"Saved: {png_path}")

    if SAVE_PDF_TOO:
        pdf_path = f"{path_base}.pdf"
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Saved: {pdf_path}")


def _maybe_add_value_labels(ax, bars, fmt="{:.1f}"):
    if not ADD_VALUE_LABELS:
        return
    # Keep labels small and only for reasonably small subplots
    for b in bars:
        h = b.get_height()
        if np.isfinite(h) and h != 0:
            ax.annotate(fmt.format(h),
                        (b.get_x() + b.get_width() / 2, h),
                        textcoords="offset points",
                        xytext=(0, 2),
                        ha="center", va="bottom",
                        fontsize=10, alpha=0.9)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 – one subplot per dataset, grouped bars per method
# ══════════════════════════════════════════════════════════════════════════════
def plot_by_dataset(rows, out_dir):
    datasets = sorted({r["dataset"] for r in rows})
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.6 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    # fig.suptitle("PPL by dataset (grouped by method)", fontsize=15, fontweight="bold", y=1.02)

    offsets, bar_w = _bar_offsets(len(PPL_SERIES))
    legend_items = [(label, color) for _, label, color in PPL_SERIES]

    for i, dataset in enumerate(datasets):
        ax = axes.flat[i]
        _setup_ax(ax)

        raw = {r["method"]: r for r in rows if r["dataset"] == dataset}
        subset = [raw[m] for m in METHOD_ORDER if m in raw]
        x = np.arange(len(subset))

        for j, (k, label, color) in enumerate(PPL_SERIES):
            vals = [r["means"][k] for r in subset]
            yerr = [r["errs"][k] for r in subset] if INCLUDE_ERROR_BARS else None

            bars = ax.bar(
                x + offsets[j], vals, bar_w,
                color=color, alpha=0.92,
                edgecolor="white", linewidth=0.6,
                yerr=yerr if (yerr and any(e > 0 for e in yerr)) else None,
                capsize=2.5 if (yerr and any(e > 0 for e in yerr)) else 0,
                zorder=3,
            )
            _maybe_add_value_labels(ax, bars)

        ax.set_title(_wrap_title(dataset), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(r["method"], r["method"]) for r in subset])
        _apply_ppl_scale(ax)

        # Only left column gets y-label (less clutter)
        if (i % ncols) == 0:
            ax.set_ylabel("Perplexity")
        else:
            ax.set_ylabel("")

    # Hide unused axes
    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)

    fig.legend(
        handles=_legend_patches(legend_items),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        fontsize=17
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    suffix = "_log" if PPL_YSCALE == "log" else ""
    _save(fig, os.path.join(out_dir, f"ppl_by_dataset{suffix}"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 – one subplot per method, grouped bars per dataset
# ══════════════════════════════════════════════════════════════════════════════
def plot_by_method(rows, out_dir):
    methods_present = sorted({r["method"] for r in rows})
    # Prefer your explicit order (and append anything unexpected at the end)
    methods = [m for m in METHOD_ORDER if m in methods_present] + [m for m in methods_present if m not in METHOD_ORDER]

    datasets = sorted({r["dataset"] for r in rows})

    ncols = 3
    nrows = int(np.ceil(len(methods) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.6 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    # fig.suptitle("PPL by method (grouped by dataset)", fontsize=15, fontweight="bold", y=1.02)

    offsets, bar_w = _bar_offsets(len(PPL_SERIES))
    legend_items = [(label, color) for _, label, color in PPL_SERIES]

    for i, method in enumerate(methods):
        ax = axes.flat[i]
        _setup_ax(ax)

        # Keep dataset ordering consistent across subplots
        by_ds = {r["dataset"]: r for r in rows if r["method"] == method}
        subset = [by_ds[d] for d in datasets if d in by_ds]
        x = np.arange(len(subset))

        for j, (k, label, color) in enumerate(PPL_SERIES):
            vals = [r["means"][k] for r in subset]
            yerr = [r["errs"][k] for r in subset] if INCLUDE_ERROR_BARS else None

            bars = ax.bar(
                x + offsets[j], vals, bar_w,
                color=color, alpha=0.92,
                edgecolor="white", linewidth=0.6,
                yerr=yerr if (yerr and any(e > 0 for e in yerr)) else None,
                capsize=2.5 if (yerr and any(e > 0 for e in yerr)) else 0,
                zorder=3,
            )
            _maybe_add_value_labels(ax, bars)

        ax.set_title(SHORT.get(method, method).replace("\n", " "), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([r["dataset"].replace("_", "\n") for r in subset])
        _apply_ppl_scale(ax)

        if (i % ncols) == 0:
            ax.set_ylabel("Perplexity")
        else:
            ax.set_ylabel("")

    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)

    fig.legend(
        handles=_legend_patches(legend_items),
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        fontsize=17
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    suffix = "_log" if PPL_YSCALE == "log" else ""
    _save(fig, os.path.join(out_dir, f"ppl_by_method{suffix}"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 – relative increase over PPL_ori per dataset
# ══════════════════════════════════════════════════════════════════════════════
def plot_relative_increase(rows, out_dir):
    datasets = sorted({r["dataset"] for r in rows})
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.6 * nrows), sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    # fig.suptitle(
    #     "Relative PPL Increase over PPL_ori  [(PPL_x − PPL_ori) / PPL_ori]",
    #     fontsize=12, fontweight="bold", y=1.02
    # )

    # As before: 3 ratio bars only
    ratio_series = RATIO_SERIES

    offsets, bar_w = _bar_offsets(len(ratio_series))
    legend_items = [(label, color) for _, label, color in ratio_series]

    for i, dataset in enumerate(datasets):
        ax = axes.flat[i]
        _setup_ax(ax)

        raw = {r["method"]: r for r in rows if r["dataset"] == dataset}
        subset = [raw[m] for m in METHOD_ORDER if m in raw]
        x = np.arange(len(subset))

        for j, (k, label, color) in enumerate(ratio_series):
            vals = []
            for r in subset:
                ori = r["means"]["ori"]
                vals.append((r["means"][k] - ori) / ori if ori > 0 else 0.0)

            bars = ax.bar(
                x + offsets[j], vals, bar_w,
                color=color, alpha=0.92,
                edgecolor="white", linewidth=0.6,
                zorder=3,
            )
            _maybe_add_value_labels(ax, bars, fmt="{:.0%}")

        ax.axhline(0, color="black", linewidth=0.9, alpha=0.85)
        ax.set_title(_wrap_title(dataset), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(r["method"], r["method"]) for r in subset], fontsize=15)

        ax.set_ylabel("Relative Increase" if (i % ncols) == 0 else "")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))

    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)

    fig.legend(
        handles=_legend_patches(legend_items),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        fontsize=17
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    _save(fig, os.path.join(out_dir, "ppl_relative_increase"))
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 – nominal (absolute) change over PPL_ori per dataset
# ══════════════════════════════════════════════════════════════════════════════
def plot_nominal_change(rows, out_dir):
    datasets = sorted({r["dataset"] for r in rows})
    ncols = 3
    nrows = int(np.ceil(len(datasets) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.6 * ncols, 4.6 * nrows), sharey=False)
    axes = np.array(axes).reshape(nrows, ncols)

    ratio_series = RATIO_SERIES

    offsets, bar_w = _bar_offsets(len(ratio_series))
    legend_items = [(label, color) for _, label, color in ratio_series]

    for i, dataset in enumerate(datasets):
        ax = axes.flat[i]
        _setup_ax(ax)

        raw = {r["method"]: r for r in rows if r["dataset"] == dataset}
        subset = [raw[m] for m in METHOD_ORDER if m in raw]
        x = np.arange(len(subset))

        for j, (k, label, color) in enumerate(ratio_series):
            vals = [r["means"][k] - r["means"]["ori"] for r in subset]

            bars = ax.bar(
                x + offsets[j], vals, bar_w,
                color=color, alpha=0.92,
                edgecolor="white", linewidth=0.6,
                zorder=3,
            )
            _maybe_add_value_labels(ax, bars, fmt="{:.1f}")

        ax.axhline(0, color="black", linewidth=0.9, alpha=0.85)
        ax.set_title(_wrap_title(dataset), fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT.get(r["method"], r["method"]) for r in subset], fontsize=15)

        ax.set_ylabel("PPL Change (absolute)" if (i % ncols) == 0 else "")
        ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis="y", style="plain")

    for ax in axes.flat[len(datasets):]:
        ax.set_visible(False)

    fig.legend(
        handles=_legend_patches(legend_items),
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.01),
        frameon=False,
        fontsize=17
    )

    fig.tight_layout(rect=[0, 0.06, 1, 0.98])
    _save(fig, os.path.join(out_dir, "ppl_nominal_change"))
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    out_dir = os.path.join(SUMMARY_FOLDER, "charts")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading PPL change ratios data...")
    change_ratios_data = load_ppl_change_ratios_stats(RESULTS)

    if not change_ratios_data:
        print("No ppl_change_ratios_stats.json files found. Check your RESULTS path in config.yaml.")
        raise SystemExit(1)

    rows = build_rows(change_ratios_data)
    print(f"Loaded {len(rows)} method/dataset combinations.\n")

    plot_by_dataset(rows, out_dir)
    plot_by_method(rows, out_dir)
    plot_relative_increase(rows, out_dir)
    plot_nominal_change(rows, out_dir)

    print(f"\nAll charts saved to: {out_dir}")