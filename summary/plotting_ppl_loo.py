"""
Script for plotting Leave-One-Out (LOO) perplexity analysis.
Two visualizations:
1. Bar chart of ppl_change per sentence position
2. Box plot of ppl_without_sentence per sentence position
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.image as mpimg
import os
import numpy as np


# Framework colors for consistent plotting
FRAMEWORK_COLORS = {
    "templated_narrative": "tab:blue",
    "xaistories_narratives": "tab:red",
    "xaistories": "tab:green",
    "explingo_narratives": "tab:orange",
    "explingo_zero_shot": "tab:purple"
}

FRAMEWORK_LABELS = {
    "templated_narrative": "Templated Explanation",
    "xaistories_narratives": "XAIstories Narratives",
    "xaistories": "XAIstories",
    "explingo_narratives": "Explingo Narratives",
    "explingo_zero_shot": "Explingo Zero-Shot"
}


def load_loo_data(path: str, dataset: str):
    """Load leave-one-out perplexity CSV file for a dataset."""
    file_path = os.path.join(path, dataset, "ppl_leave_one_out.csv")
    df = pd.read_csv(file_path)
    # Make sentence_index 1-indexed for display
    df["sentence_position"] = df["sentence_index"] + 1
    return df


def plot_ppl_change_bar(base_results_path: str, dataset: str, frameworks: list,
                        save_path: str = None, show: bool = True):
    """
    Bar chart showing average ppl_change per sentence position.
    Positive ppl_change = removing that sentence increases PPL = sentence was important for coherence.
    """
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    
    bar_width = 0.8 / len(frameworks)
    positions_set = set()
    stats_text = []  # Collect stats for annotation
    
    for i, framework in enumerate(frameworks):
        results_path = os.path.join(base_results_path, framework)
        dataset_path = os.path.join(results_path, dataset)
        
        if not os.path.exists(dataset_path):
            print(f"  Skipping {framework}/{dataset} (not found)")
            continue
        
        try:
            df = load_loo_data(results_path, dataset)
            
            # Compute overall mean and std of ppl_change for this framework
            overall_mean = df["ppl_change"].mean()
            overall_std = df["ppl_change"].std()
            
            # Compute mean ppl_change per sentence position
            mean_change = df.groupby("sentence_position")["ppl_change"].mean()
            positions = mean_change.index.values
            positions_set.update(positions)
            
            color = FRAMEWORK_COLORS.get(framework, "tab:gray")
            label = FRAMEWORK_LABELS.get(framework, framework)
            
            # Add stats to label
            label_with_stats = f"{label} (μ={overall_mean:.2f}, σ={overall_std:.2f})"
            stats_text.append(f"{label}: μ={overall_mean:.2f}, σ={overall_std:.2f}")
            
            # Offset bars for each framework
            x_positions = positions + (i - len(frameworks)/2 + 0.5) * bar_width
            
            plt.bar(x_positions, mean_change.values, width=bar_width, 
                    color=color, alpha=0.8, label=label_with_stats, edgecolor='black', linewidth=0.5)
            
        except Exception as e:
            print(f"  ERROR loading {framework}/{dataset}: {e}")
    
    # Add horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    plt.title(f"PPL Change When Removing Sentence: {dataset.replace('_', ' ').title()}", fontsize=20)
    plt.xlabel("Sentence Position", fontsize=16)
    plt.ylabel("Mean PPL Change (Δ)", fontsize=16)
    plt.grid(True, linestyle="--", linewidth=0.5, axis='y')
    plt.legend(fontsize=11, loc='upper right')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_ppl_distributions(base_results_path: str, dataset: str, frameworks: list,
                           save_path: str = None, show: bool = True):
    """
    Box plot comparing Original PPL vs LOO PPL distributions per sentence position.
    For each sentence position, shows side-by-side: Original PPL and PPL without that sentence.
    """
    fig, axes = plt.subplots(1, len(frameworks), figsize=(8*len(frameworks), 7), sharey=True)
    
    if len(frameworks) == 1:
        axes = [axes]
    
    for ax, framework in zip(axes, frameworks):
        results_path = os.path.join(base_results_path, framework)
        dataset_path = os.path.join(results_path, dataset)
        
        if not os.path.exists(dataset_path):
            ax.text(0.5, 0.5, f'{framework}\nnot found', ha='center', va='center')
            continue
        
        try:
            df = load_loo_data(results_path, dataset)
            
            # Get unique sentence positions
            sentence_positions = sorted(df["sentence_position"].unique())
            
            color = FRAMEWORK_COLORS.get(framework, "tab:gray")
            label = FRAMEWORK_LABELS.get(framework, framework)
            
            # Prepare data for grouped boxplots
            # For each sentence position: [original_ppl, loo_ppl]
            data = []
            positions = []
            colors = []
            width = 0.35
            
            for i, sent_pos in enumerate(sentence_positions):
                sent_df = df[df["sentence_position"] == sent_pos]
                
                # Original PPL for texts that have this sentence position
                original_ppl = sent_df["original_ppl"].values
                # LOO PPL when this sentence is removed
                loo_ppl = sent_df["ppl_without_sentence"].values
                
                data.append(original_ppl)
                data.append(loo_ppl)
                
                # Position: original slightly left, LOO slightly right
                base_pos = sent_pos
                positions.append(base_pos - width/2)
                positions.append(base_pos + width/2)
                
                colors.append(color)  # Original
                colors.append('lightgray')  # LOO
            
            bp = ax.boxplot(data, positions=positions, widths=width*0.8, patch_artist=True)
            
            # Color the boxes alternating: framework color for original, gray for LOO
            for idx, patch in enumerate(bp['boxes']):
                patch.set_facecolor(colors[idx])
                patch.set_alpha(0.7)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)
            
            ax.set_title(label, fontsize=16)
            ax.set_xlabel("Sentence Position", fontsize=14)
            ax.set_xticks(sentence_positions)
            ax.set_xticklabels([str(p) for p in sentence_positions], fontsize=12)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True, linestyle="--", linewidth=0.5, axis='y')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color, alpha=0.7, label='Original PPL'),
                Patch(facecolor='lightgray', alpha=0.7, label='LOO PPL')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'ERROR: {e}', ha='center', va='center')
    
    axes[0].set_ylabel("Perplexity", fontsize=14)
    
    dataset_title = "German Credit" if dataset == "german_credit" else dataset.replace('_', ' ').title()
    fig.suptitle(f"Original vs LOO Perplexity per Sentence: {dataset_title}", 
                 fontsize=20, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_combined_bar_chart(charts_path: str, output_dir: str, datasets: list):
    """Create a combined chart from individual bar chart plots."""
    chart_files = [os.path.join(charts_path, f'loo_ppl_change_{ds}.png') for ds in datasets]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    for idx, (ax, chart_file) in enumerate(zip(axes, chart_files)):
        if os.path.exists(chart_file):
            img = mpimg.imread(chart_file)
            ax.imshow(img)
            ax.axis('off')
            ax.text(0.02, 1.01, f'({chr(97+idx)})', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')
        else:
            ax.text(0.5, 0.5, f'Chart {idx+1}\nnot found', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')

    fig.suptitle("LOO PPL Change by Sentence Position", fontsize=25, y=.93)
    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'combined_loo_bar_charts.png')
    pdf_path = os.path.join(output_dir, 'combined_loo_bar_charts.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Combined chart saved as {png_path} and {pdf_path}")


def create_combined_boxplot(charts_path: str, output_dir: str, datasets: list):
    """Create a combined chart from individual box plot charts."""
    chart_files = [os.path.join(charts_path, f'loo_boxplot_{ds}.png') for ds in datasets]

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))
    axes = axes.flatten()

    for idx, (ax, chart_file) in enumerate(zip(axes, chart_files)):
        if os.path.exists(chart_file):
            img = mpimg.imread(chart_file)
            ax.imshow(img)
            ax.axis('off')
            ax.text(0.02, 1.01, f'({chr(97+idx)})', transform=ax.transAxes,
                    fontsize=14, fontweight='bold', va='top')
        else:
            ax.text(0.5, 0.5, f'Chart {idx+1}\nnot found', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')

    fig.suptitle("Original vs LOO Perplexity Distribution", fontsize=25, y=.93)
    plt.subplots_adjust(hspace=0.05)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'combined_loo_boxplot_charts.png')
    pdf_path = os.path.join(output_dir, 'combined_loo_boxplot_charts.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Combined chart saved as {png_path} and {pdf_path}")


def main():
    # Configuration
    BASE_RESULTS_PATH = "/home/mcedro/phd/repos/xain_xaid/XAIN_XAID/results"
    OUTPUT_DIR = "/home/mcedro/phd/repos/xain_xaid/XAIN_XAID/summary/plots/loo"
    DATASETS = ["compas", "diabetes", "fifa", "german_credit", "stroke", "student"]
    FRAMEWORKS = [
        "templated_narrative",
        "xaistories"
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. Bar charts: PPL Change per sentence position ---
    print("="*60)
    print("Generating Bar Charts (PPL Change)")
    print("="*60)
    for dataset in DATASETS:
        print(f"\nProcessing {dataset}...")
        save_path = os.path.join(OUTPUT_DIR, f"loo_ppl_change_{dataset}.png")
        plot_ppl_change_bar(BASE_RESULTS_PATH, dataset, FRAMEWORKS,
                            save_path=save_path, show=False)

    print("\nCreating combined bar chart...")
    create_combined_bar_chart(OUTPUT_DIR, OUTPUT_DIR, DATASETS)

    # --- 2. Box plots: Original PPL vs LOO PPL distribution ---
    print("\n" + "="*60)
    print("Generating Box Plots (Original vs LOO PPL Distribution)")
    print("="*60)
    for dataset in DATASETS:
        print(f"\nProcessing {dataset}...")
        save_path = os.path.join(OUTPUT_DIR, f"loo_boxplot_{dataset}.png")
        plot_ppl_distributions(BASE_RESULTS_PATH, dataset, FRAMEWORKS,
                               save_path=save_path, show=False)

    print("\nCreating combined boxplot chart...")
    create_combined_boxplot(OUTPUT_DIR, OUTPUT_DIR, DATASETS)

    print("\nDone!")


if __name__ == "__main__":
    main()
