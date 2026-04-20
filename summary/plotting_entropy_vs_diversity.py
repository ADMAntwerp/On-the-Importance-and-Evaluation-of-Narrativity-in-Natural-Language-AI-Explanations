"""
Script for plotting Entropy (ln(PPL)) vs Lexical Diversity (Dist2) scatter plot.

X-axis: Entropy (ln(PPL)) — Proxy for "Predictability"
Y-axis: Lexical Diversity (Dist2) — Proxy for "Information Content"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from pathlib import Path


# Load configuration
with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    RESULTS = config_data["RESULTS"]
    SUMMARY_FOLDER = config_data["SUMMARY_FOLDER"]


# Framework colors for consistent plotting
FRAMEWORK_COLORS = {
    "talktomodel": "tab:brown",
    "templated_narrative": "tab:blue",
    "explingo": "tab:cyan",
    "explingo_zero_shot": "tab:green",
    "explingo_narratives": "tab:orange",
    "xaistories": "tab:red",
    "xaistories_narratives": "tab:purple",
}

FRAMEWORK_LABELS = {
    "templated_narrative": "Templated Expl.",
    "explingo_narratives": "Explingo (Narr.)",
    "explingo_zero_shot": "Explingo (Z-S)",
    "xaistories": "XAIstories",
    "xaistories_narratives": "XAIstories (Narr.)",
    "explingo": "Explingo",
    "talktomodel": "TalkToModel"
}

DATASET_MARKERS = {
    "dataset1": "o",
    "dataset2": "s",
    "dataset3": "^",
    "dataset4": "D",
    "dataset5": "P",
    "dataset6": "X",
}


def load_cem_data(results_folder: str) -> pd.DataFrame:
    """
    Load cem_detailed.csv from all methods and datasets.
    Returns a DataFrame with columns: method, dataset, perplexity, ln_perplexity, distinct_bigram_ratio
    """
    all_data = []
    
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
            cem_path = os.path.join(method_path, dataset, "cem_detailed.csv")
            if os.path.exists(cem_path):
                try:
                    df = pd.read_csv(cem_path)
                    df["method"] = method
                    df["dataset"] = dataset
                    # Compute ln(perplexity) if not already present
                    if "ln_perplexity" not in df.columns:
                        df["ln_perplexity"] = np.log(df["perplexity"])
                    all_data.append(df)
                except Exception as e:
                    print(f"Error loading {cem_path}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()


def aggregate_by_method_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data to get mean and std per method-dataset combination."""
    agg_df = df.groupby(["method", "dataset"]).agg({
        "ln_perplexity": ["mean", "std"],
        "distinct_bigram_ratio": ["mean", "std"]
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = [
        "method", "dataset",
        "entropy_mean", "entropy_std",
        "dist2_mean", "dist2_std"
    ]
    return agg_df


def aggregate_by_method(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data to get mean and std per method (across all datasets)."""
    agg_df = df.groupby("method").agg({
        "ln_perplexity": ["mean", "std"],
        "distinct_bigram_ratio": ["mean", "std"]
    }).reset_index()
    
    # Flatten column names
    agg_df.columns = [
        "method",
        "entropy_mean", "entropy_std",
        "dist2_mean", "dist2_std"
    ]
    return agg_df


def plot_entropy_vs_diversity_by_method(
    results_folder: str = RESULTS,
    save_path: str = None,
    show: bool = True,
    aggregate: str = "method"  # "method" or "method_dataset" or "instance"
):
    """
    Create scatter plot: Entropy (ln(PPL)) vs Lexical Diversity (Dist2).
    
    Args:
        results_folder: Path to results folder
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
        aggregate: Level of aggregation - "method", "method_dataset", or "instance"
    """
    # Load data
    df = load_cem_data(results_folder)
    if df.empty:
        print("No data found!")
        return
    
    # Remove rows with NaN values
    df = df.dropna(subset=["perplexity", "distinct_bigram_ratio"])
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    if aggregate == "instance":
        # Plot individual instances
        for method in FRAMEWORK_COLORS.keys():
            if method not in df["method"].unique():
                continue
            method_df = df[df["method"] == method]
            color = FRAMEWORK_COLORS.get(method, "tab:gray")
            label = FRAMEWORK_LABELS.get(method, method)
            
            ax.scatter(
                method_df["ln_perplexity"],
                method_df["distinct_bigram_ratio"],
                c=color,
                marker="o",
                label=label,
                alpha=0.5,
                s=30,
                edgecolors="none"
            )
    
    elif aggregate == "method_dataset":
        # Plot individual instances (faint)
        for method in FRAMEWORK_COLORS.keys():
            if method not in df["method"].unique():
                continue
            method_df = df[df["method"] == method]
            color = FRAMEWORK_COLORS.get(method, "tab:gray")
            
            ax.scatter(
                method_df["ln_perplexity"],
                method_df["distinct_bigram_ratio"],
                c=color,
                marker="o",
                alpha=0.15,
                s=20,
                edgecolors="none"
            )
        
        # Aggregate by method-dataset
        agg_df = aggregate_by_method_dataset(df)
        
        for method in FRAMEWORK_COLORS.keys():
            if method not in agg_df["method"].unique():
                continue
            method_df = agg_df[agg_df["method"] == method]
            color = FRAMEWORK_COLORS.get(method, "tab:gray")
            label = FRAMEWORK_LABELS.get(method, method)
            
            ax.errorbar(
                method_df["entropy_mean"],
                method_df["dist2_mean"],
                xerr=method_df["entropy_std"],
                yerr=method_df["dist2_std"],
                fmt="o",
                color=color,
                label=label,
                markersize=10,
                capsize=3,
                alpha=0.8,
                elinewidth=1
            )   
    
    else:  # aggregate == "method"
        # Aggregate by method only
        agg_df = aggregate_by_method(df)
        
        for method in FRAMEWORK_COLORS.keys():
            if method not in agg_df["method"].values:
                continue
            row = agg_df[agg_df["method"] == method].iloc[0]
            color = FRAMEWORK_COLORS.get(method, "tab:gray")
            label = FRAMEWORK_LABELS.get(method, method)
            
            ax.errorbar(
                row["entropy_mean"],
                row["dist2_mean"],
                xerr=row["entropy_std"],
                yerr=row["dist2_std"],
                fmt="o",
                color=color,
                label=label,
                markersize=14,
                capsize=5,
                alpha=0.9,
                elinewidth=2,
                markeredgewidth=2,
                markeredgecolor="white"
            )
    
    # Styling
    ax.set_xlabel(r"Entropy ($\mathit{ln}(\mathrm{PPL})$)", fontsize=17)
    ax.set_ylabel(r"Lexical Diversity (Dist$_2$)", fontsize=17)
    # ax.set_title("Entropy vs Lexical Diversity", fontsize=18, pad=15)
    ax.tick_params(axis="both", which="major", labelsize=15)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax.set_xlim(1.2, 3.42)
    ax.set_ylim(0.5, 1.01)
    
    # Improved legend
    legend = ax.legend(
        loc="lower right",
        fontsize=17,
        framealpha=0.95,
        edgecolor="gray",
        fancybox=True,
        shadow=False,
        ncol=1,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.8
    )
    legend.get_frame().set_linewidth(0.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()
    return fig


def plot_entropy_vs_diversity_combined(
    results_folder: str = RESULTS,
    save_path: str = None,
    show: bool = True
):
    """
    Create a combined scatter plot showing both:
    - Individual instances (faint markers)
    - Aggregated means per method (bold markers with error bars)
    """
    # Load data
    df = load_cem_data(results_folder)
    if df.empty:
        print("No data found!")
        return
    
    # Remove rows with NaN values
    df = df.dropna(subset=["perplexity", "distinct_bigram_ratio"])
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Get aggregated data
    agg_df = aggregate_by_method(df)
    
    # Plot individual instances (faint)
    for method in FRAMEWORK_COLORS.keys():
        if method not in df["method"].unique():
            continue
        method_df = df[df["method"] == method]
        color = FRAMEWORK_COLORS.get(method, "tab:gray")
        
        ax.scatter(
            method_df["ln_perplexity"],
            method_df["distinct_bigram_ratio"],
            c=color,
            marker="o",
            alpha=0.15,
            s=20,
            edgecolors="none"
        )
    
    # Plot aggregated means with error bars (bold)
    for method in FRAMEWORK_COLORS.keys():
        if method not in agg_df["method"].values:
            continue
        row = agg_df[agg_df["method"] == method].iloc[0]
        color = FRAMEWORK_COLORS.get(method, "tab:gray")
        label = FRAMEWORK_LABELS.get(method, method)
        
        ax.errorbar(
            row["entropy_mean"],
            row["dist2_mean"],
            xerr=row["entropy_std"],
            yerr=row["dist2_std"],
            fmt="o",
            color=color,
            label=label,
            markersize=14,
            capsize=5,
            alpha=0.95,
            elinewidth=2,
            markeredgewidth=2,
            markeredgecolor="white",
            zorder=10
        )
    
    # Styling
    ax.set_xlabel(r"Entropy $\ln(\mathrm{PPL})$ — Predictability Proxy", fontsize=16)
    ax.set_ylabel(r"Lexical Diversity (Dist2) — Information Content Proxy", fontsize=16)
    ax.set_title("Entropy vs Lexical Diversity", fontsize=18, pad=15)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
    ax.set_xlim(1.2, 3.52)
    ax.set_ylim(0.5, 1.01)
    
    # Improved legend
    legend = ax.legend(
        loc="lower right",
        fontsize=19,
        framealpha=0.95,
        edgecolor="gray",
        fancybox=True,
        shadow=False,
        ncol=1,
        columnspacing=1.0,
        handletextpad=0.5,
        borderpad=0.8
    )
    legend.get_frame().set_linewidth(0.5)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    
    plt.close()
    return fig


def plot_entropy_vs_diversity_single_dataset(
    results_folder: str,
    dataset: str,
    save_path: str = None,
    show: bool = True
):
    """
    Create a single scatter plot for one dataset.
    Shows Entropy (ln(PPL)) vs Lexical Diversity (Dist2) per method.
    """
    # Load data
    df = load_cem_data(results_folder)
    if df.empty:
        print("No data found!")
        return
    
    # Remove rows with NaN values and filter to dataset
    df = df.dropna(subset=["perplexity", "distinct_bigram_ratio"])
    dataset_df = df[df["dataset"] == dataset]
    
    if dataset_df.empty:
        print(f"No data for dataset: {dataset}")
        return
    
    plt.figure(figsize=(9, 7))
    ax = plt.gca()
    
    # Aggregate by method for this dataset
    agg_df = dataset_df.groupby("method").agg({
        "ln_perplexity": ["mean", "std"],
        "distinct_bigram_ratio": ["mean", "std"]
    }).reset_index()
    agg_df.columns = ["method", "entropy_mean", "entropy_std", "dist2_mean", "dist2_std"]
    
    # Plot each method
    legend_handles = []
    legend_labels = []
    for method in FRAMEWORK_COLORS.keys():
        if method not in agg_df["method"].values:
            continue
        
        color = FRAMEWORK_COLORS.get(method, "tab:gray")
        label = FRAMEWORK_LABELS.get(method, method)
        
        # Plot individual datapoints (faint)
        method_df = dataset_df[dataset_df["method"] == method]
        ax.scatter(
            method_df["ln_perplexity"],
            method_df["distinct_bigram_ratio"],
            c=color,
            marker="o",
            alpha=0.15,
            s=30,
            edgecolors="none"
        )
        
        # Plot aggregated mean with error bars (bold)
        row = agg_df[agg_df["method"] == method].iloc[0]
        ax.errorbar(
            row["entropy_mean"],
            row["dist2_mean"],
            xerr=row["entropy_std"],
            yerr=row["dist2_std"],
            fmt="o",
            color=color,
            markersize=12,
            capsize=5,
            alpha=0.9,
            elinewidth=2,
            markeredgewidth=2,
            markeredgecolor="white",
            zorder=10
        )
        
        # Create custom legend handle (dot only, no error bars)
        from matplotlib.lines import Line2D
        handle = Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                        markersize=10, markeredgecolor='white', markeredgewidth=1.5)
        legend_handles.append(handle)
        legend_labels.append(label)
    
    # Styling aligned with plotting_ppl.py
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.title(f"{dataset.replace('_', ' ').title()}", fontsize=26, y=1.03)
    plt.xlabel(r"Entropy [$\ln(\mathrm{PPL})]$", fontsize=23)
    plt.ylabel("Lexical Diversity (Distinct-2 Ratio)", fontsize=23)
    plt.grid(True, linestyle="--", linewidth=0.5)
    ax.set_xlim(1.2, 3.52)
    ax.set_ylim(0.5, 1.01)
    plt.legend(legend_handles, legend_labels, fontsize=16, loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_combined_entropy_diversity_chart(charts_path: str, output_dir: str, datasets: list):
    """Create a combined chart from individual dataset plots (2x3 grid)."""
    import matplotlib.image as mpimg
    
    chart_files = [os.path.join(charts_path, f'entropy_vs_diversity_{ds}.png') for ds in datasets]

    # Create a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Load and display each chart
    for idx, (ax, chart_file) in enumerate(zip(axes, chart_files)):
        if os.path.exists(chart_file):
            img = mpimg.imread(chart_file)
            ax.imshow(img)
            ax.axis('off')
            # Add subplot labels (a), (b), (c), etc.
            ax.text(0.02, 1.01, f'({chr(97+idx)})', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')
        else:
            ax.text(0.5, 0.5, f'Chart {idx+1}\nnot found', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')

    fig.suptitle("Entropy-Diversity Profiles of Natural Language XAI Explanations", fontsize=29, y=.93)
    plt.subplots_adjust(hspace=-0.2)
                        
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'entropy_vs_diversity_grid.png')
    pdf_path = os.path.join(output_dir, 'entropy_vs_diversity_grid.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    
    if False:  # Don't show by default
        plt.show()
    else:
        plt.close()

    print(f"Combined chart saved as {png_path} and {pdf_path}")


if __name__ == "__main__":
    # Output directory
    output_dir = os.path.join(SUMMARY_FOLDER, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Aggregated by method with error bars
    print("Creating plot: Entropy vs Diversity (aggregated by method)...")
    plot_entropy_vs_diversity_by_method(
        results_folder=RESULTS,
        save_path=os.path.join(output_dir, "entropy_vs_diversity_by_method.png"),
        show=False,
        aggregate="method"
    )
    plot_entropy_vs_diversity_by_method(
        results_folder=RESULTS,
        save_path=os.path.join(output_dir, "entropy_vs_diversity_by_method.pdf"),
        show=False,
        aggregate="method"
    )
    
    # Plot 2: Combined view (instances + aggregated means)
    print("Creating plot: Entropy vs Diversity (combined view)...")
    plot_entropy_vs_diversity_combined(
        results_folder=RESULTS,
        save_path=os.path.join(output_dir, "entropy_vs_diversity_combined.png"),
        show=False
    )
    plot_entropy_vs_diversity_combined(
        results_folder=RESULTS,
        save_path=os.path.join(output_dir, "entropy_vs_diversity_combined.pdf"),
        show=False
    )
    
    # Plot 3: Aggregated by method-dataset
    print("Creating plot: Entropy vs Diversity (by method-dataset)...")
    plot_entropy_vs_diversity_by_method(
        results_folder=RESULTS,
        save_path=os.path.join(output_dir, "entropy_vs_diversity_by_method_dataset.png"),
        show=False,
        aggregate="method_dataset"
    )
    plot_entropy_vs_diversity_by_method(
        results_folder=RESULTS,
        save_path=os.path.join(output_dir, "entropy_vs_diversity_by_method_dataset.pdf"),
        show=False,
        aggregate="method_dataset"
    )
    
    # Plot 4: Individual plots per dataset (like plotting_ppl.py)
    DATASETS = ["compas", "diabetes", "fifa", "german_credit", "stroke", "student"]
    print("\nGenerating individual plots per dataset...")
    for dataset in DATASETS:
        print(f"  Processing {dataset}...")
        save_path = os.path.join(output_dir, f"entropy_vs_diversity_{dataset}.png")
        plot_entropy_vs_diversity_single_dataset(
            results_folder=RESULTS,
            dataset=dataset,
            save_path=save_path,
            show=False
        )
    
    # Plot 5: 2x3 grid combining all datasets
    print("\nCreating combined 2x3 grid chart...")
    create_combined_entropy_diversity_chart(output_dir, output_dir, DATASETS)
    
    print("\nAll plots saved to:", output_dir)
