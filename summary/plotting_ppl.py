"""
Script for plotting cumulative perplexity by sentence for multiple datasets and frameworks.
Combines all frameworks on a single plot per dataset.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.image as mpimg
import os
from pathlib import Path
import ast


# Framework colors for consistent plotting
FRAMEWORK_COLORS = {
    "templated_narrative": "tab:blue",
    "explingo_narratives": "tab:orange",
    "explingo_zero_shot": "tab:green",
    "xaistories": "tab:red",
    "xaistories_narratives": "tab:purple"
}

FRAMEWORK_LABELS = {
    "templated_narrative": "Templated Expl.",
    "explingo_narratives": "Explingo Narr.",
    "explingo_zero_shot": "Explingo Z-S",
    "xaistories": "XAIstories",
    "xaistories_narratives": "XAIstories Narr."
}


def parse_cumulative_ppls(df: pd.DataFrame) -> pd.DataFrame:
    """Parse cumulative_ppls column from string list to expanded rows."""
    rows = []
    for _, row in df.iterrows():
        text_id = row["text_id"]
        ppls = ast.literal_eval(row["cumulative_ppls"])
        for i, ppl in enumerate(ppls, start=1):
            rows.append({
                "text_id": text_id,
                "num_sentences": i,
                "ppl": ppl
            })
    return pd.DataFrame(rows)


def load_data(path: str, dataset: str):
    """Load cumulative perplexity CSV file for a dataset."""
    ori_path = os.path.join(path, dataset, "cumulative_ppl_detailed.csv")
    df = pd.read_csv(ori_path)
    df = parse_cumulative_ppls(df)
    df = df.sort_values(["text_id", "num_sentences"])
    return df


def compute_mean_curve(df):
    """Compute mean curve averaging across text_ids per sentence index."""
    return df.groupby("num_sentences", as_index=False)["ppl"].mean()


def plot_combined_frameworks(base_results_path: str, dataset: str, frameworks: list,
                              save_path: str = None, show: bool = True):
    """Plot cumulative perplexity for all frameworks on a single plot."""
    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    for framework in frameworks:
        results_path = os.path.join(base_results_path, framework)
        dataset_path = os.path.join(results_path, dataset)
        
        if not os.path.exists(dataset_path):
            print(f"  Skipping {framework}/{dataset} (not found)")
            continue
        
        try:
            df = load_data(results_path, dataset)
            mean_curve = compute_mean_curve(df)
            color = FRAMEWORK_COLORS.get(framework, "tab:gray")
            label = FRAMEWORK_LABELS.get(framework, framework)
            
            # Faint per-sequence lines
            for tid, g in df.groupby("text_id"):
                plt.plot(g["num_sentences"], g["ppl"], linewidth=0.8, alpha=0.15, color=color)
            
            # Strong mean line
            plt.plot(mean_curve["num_sentences"], mean_curve["ppl"],
                     linewidth=3.0, alpha=1.0, color=color, label=label, zorder=5)
        except Exception as e:
            print(f"  ERROR loading {framework}/{dataset}: {e}")

    # Make y start at 0 and x ticks integers
    plt.ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=16)

    plt.title(f"{dataset.replace('_', ' ').title()}", fontsize=26, fontweight='bold', y=1.03)
    plt.xlabel("Sentence index", fontsize=23)
    plt.ylabel("Cumulative perplexity", fontsize=23)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=19)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_combined_chart(charts_path: str, output_dir: str, datasets: list):
    """Create a combined chart from individual dataset plots."""
    chart_files = [os.path.join(charts_path, f'cumulative_ppl_{ds}.png') for ds in datasets]

    # Create a 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Load and display each chart
    for idx, (ax, chart_file) in enumerate(zip(axes, chart_files)):
        if os.path.exists(chart_file):
            img = mpimg.imread(chart_file)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.text(0.5, 0.5, f'Chart {idx+1}\nnot found', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')

    # fig.suptitle("Cumulative Perplexity by Sentence Across Frameworks", fontsize=25, y=.93)
    plt.subplots_adjust(hspace=-0.2), #wspace=-0.05)
                        
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'combined_frameworks_charts.png')
    pdf_path = os.path.join(output_dir, 'combined_frameworks_charts.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Combined chart saved as {png_path} and {pdf_path}")


def main():
    # Configuration
    BASE_RESULTS_PATH = "/home/mcedro/phd/repos/xain_xaid/XAIN_XAID/results"
    OUTPUT_DIR = "/home/mcedro/phd/repos/xain_xaid/XAIN_XAID/summary/plots"
    DATASETS = ["compas", "diabetes", "fifa", "german_credit", "stroke", "student"]
    FRAMEWORKS = [
        "templated_narrative",
        "explingo_narratives",
        "explingo_zero_shot",
        "xaistories",
        "xaistories_narratives"
    ]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate combined plots for each dataset (all frameworks on one plot)
    print("Generating combined framework plots per dataset...")
    for dataset in DATASETS:
        print(f"\nProcessing {dataset}...")
        save_path = os.path.join(OUTPUT_DIR, f"cumulative_ppl_{dataset}.png")
        plot_combined_frameworks(BASE_RESULTS_PATH, dataset, FRAMEWORKS,
                                  save_path=save_path, show=False)

    # Create combined chart of all datasets
    print("\nCreating combined chart of all datasets...")
    create_combined_chart(OUTPUT_DIR, OUTPUT_DIR, DATASETS)

    print("\nDone!")


if __name__ == "__main__":
    main()
