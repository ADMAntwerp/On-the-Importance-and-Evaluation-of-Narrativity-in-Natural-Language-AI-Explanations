"""
Script for plotting the distribution of PPL change when each sentence is removed.
Shows boxplots of ppl_change per sentence position for each framework.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
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
    "templated_narrative": "Templated Expl.",
    "xaistories_narratives": "XAIstories Narr.",
    "xaistories": "XAIstories",
    "explingo_narratives": "Explingo Narr.",
    "explingo_zero_shot": "Explingo Z-S"
}


def load_loo_data(path: str, dataset: str):
    """Load leave-one-out perplexity CSV file for a dataset."""
    file_path = os.path.join(path, dataset, "ppl_leave_one_out.csv")
    df = pd.read_csv(file_path)
    # Make sentence_index 1-indexed for display
    df["sentence_position"] = df["sentence_index"] + 1
    return df


def plot_ppl_change_distribution(base_results_path: str, dataset: str, frameworks: list,
                                  save_path: str = None, show: bool = True,
                                  show_ylabel: bool = True):
    """
    Box plot showing the distribution of ppl_change per sentence position.
    ppl_change = ppl_without_sentence - original_ppl
    Positive = removing sentence increases PPL (sentence was important)
    Negative = removing sentence decreases PPL (sentence hurt coherence)
    """
    fig, axes = plt.subplots(1, len(frameworks), figsize=(7*len(frameworks), 8), sharey=True)
    
    if len(frameworks) == 1:
        axes = [axes]
    
    for ax, framework in zip(axes, frameworks):
        results_path = os.path.join(base_results_path, framework)
        dataset_path = os.path.join(results_path, dataset)
        
        if not os.path.exists(dataset_path):
            ax.text(0.5, 0.5, f'{framework}\nnot found', ha='center', va='center', fontsize=22)
            continue
        
        try:
            df = load_loo_data(results_path, dataset)
            
            # Get unique sentence positions
            positions = sorted(df["sentence_position"].unique())
            
            # Prepare data for boxplot - ppl_change per position
            data = [df[df["sentence_position"] == pos]["ppl_change"].values 
                    for pos in positions]
            
            color = FRAMEWORK_COLORS.get(framework, "tab:gray")
            label = FRAMEWORK_LABELS.get(framework, framework)
            
            bp = ax.boxplot(data, positions=positions, widths=0.6, patch_artist=True)
            
            # Color the boxes
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(2)
            
            # Add horizontal line at 0
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            
            ax.set_title(label, fontsize=22)
            ax.set_xlabel("Sentence Position", fontsize=23)
            ax.set_xticks(positions)
            ax.set_xticklabels([str(p) for p in positions], fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=16)
            ax.grid(True, linestyle="--", linewidth=0.5, axis='y')
            
            # Add stats in legend
            all_changes = df["ppl_change"].values
            mean_change = np.mean(all_changes)
            std_change = np.std(all_changes)
            legend_elements = [
                Patch(facecolor=color, alpha=0.7, 
                      label=f'μ={mean_change:.2f}, σ={std_change:.2f}')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=23)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'ERROR: {e}', ha='center', va='center')
    
    if show_ylabel:
        axes[0].set_ylabel("ΔPPL (LOO - Original)", fontsize=23)
    
    dataset_title = "German Credit" if dataset == "german_credit" else dataset.replace('_', ' ').title()
    fig.suptitle(f"{dataset_title}", 
                 fontsize=26, fontweight='bold', y=.96)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_combined_chart(charts_path: str, output_dir: str, datasets: list, frameworks: list):
    """Create a combined chart from individual plots."""
    # Use _noy (no y-label) variants for the right column (odd indices)
    chart_files = []
    for idx, ds in enumerate(datasets):
        if idx % 2 == 1:  # right column
            chart_files.append(os.path.join(charts_path, f'ppl_change_dist_{ds}_noy.png'))
        else:  # left column
            chart_files.append(os.path.join(charts_path, f'ppl_change_dist_{ds}.png'))

    # Read one image to get its aspect ratio
    sample_img = mpimg.imread(chart_files[0])
    img_h, img_w = sample_img.shape[:2]
    img_aspect = img_h / img_w  # height / width of each image

    ncols, nrows = 2, 3
    subplot_width = 16 / ncols  # width per subplot in inches
    subplot_height = subplot_width * img_aspect  # preserve image aspect
    fig_height = nrows * subplot_height

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, fig_height))
    axes = axes.flatten()

    for idx, (ax, chart_file) in enumerate(zip(axes, chart_files)):
        if os.path.exists(chart_file):
            img = mpimg.imread(chart_file)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Chart {idx+1}\nnot found', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')

    framework_names = [FRAMEWORK_LABELS.get(f, f) for f in frameworks]
    #title = "Change in Perplexity Distribution under Sentence Ablation"
    #subtitle = " vs. ".join(framework_names)
    #fig.suptitle(title, fontsize=20, y=.76)
    #fig.text(0.5, 0.72, subtitle, ha='center', fontsize=18)
    plt.subplots_adjust(hspace=0.02, wspace=0.02, left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'combined_ppl_change_distribution.png')
    pdf_path = os.path.join(output_dir, 'combined_ppl_change_distribution.pdf')
    
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
        "xaistories_narratives",
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate individual plots
    print("="*60)
    print("Generating PPL Change Distribution Plots")
    print("="*60)
    for dataset in DATASETS:
        print(f"\nProcessing {dataset}...")
        save_path = os.path.join(OUTPUT_DIR, f"ppl_change_dist_{dataset}.png")
        plot_ppl_change_distribution(BASE_RESULTS_PATH, dataset, FRAMEWORKS,
                                      save_path=save_path, show=False)
        # Generate version without y-label for right column of combined chart
        save_path_noy = os.path.join(OUTPUT_DIR, f"ppl_change_dist_{dataset}_noy.png")
        plot_ppl_change_distribution(BASE_RESULTS_PATH, dataset, FRAMEWORKS,
                                      save_path=save_path_noy, show=False,
                                      show_ylabel=False)

    # Create combined chart
    print("\nCreating combined chart...")
    create_combined_chart(OUTPUT_DIR, OUTPUT_DIR, DATASETS, FRAMEWORKS)

    print("\nDone!")


if __name__ == "__main__":
    main()
