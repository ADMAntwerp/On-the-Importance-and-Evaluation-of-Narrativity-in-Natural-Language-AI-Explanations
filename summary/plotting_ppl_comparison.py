"""
Script for plotting cumulative perplexity comparing XAIstories Narratives vs Templated Explanation.
Includes original, reversed, shuffled, and leave-one-out variants.
Generates one plot per dataset with both frameworks.
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.image as mpimg
import os
import ast


# Framework colors for consistent plotting
FRAMEWORK_COLORS = {
    "templated_narrative": "tab:blue",
    "xaistories_narratives": "tab:red"
    # "xaistories": "tab:red"
}

FRAMEWORK_LABELS = {
    "templated_narrative": "Templated Expl.",
    "xaistories_narratives": "XAIstories Narr."
    #"xaistories": "XAIstories"
}

# Line styles for different variants
VARIANT_STYLES = {
    "original": {"linestyle": "-", "label_suffix": ""},
    "reversed": {"linestyle": "--", "label_suffix": " (REV)"},
    "shuffled": {"linestyle": ":", "label_suffix": " (SHUFF)"},
    "loo": {"linestyle": "-.", "label_suffix": " (LOO)"}
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


def load_data(path: str, dataset: str, variant: str = "original"):
    """Load cumulative perplexity CSV file for a dataset and variant."""
    if variant == "original":
        file_name = "ppl_cumulative.csv"
        df = pd.read_csv(os.path.join(path, dataset, file_name))
    elif variant == "reversed":
        file_name = "ppl_cumulative_rev.csv"
        df = pd.read_csv(os.path.join(path, dataset, file_name))
        df = df.rename(columns={
            "text_id_rev": "text_id",
            "num_sentences_rev": "num_sentences",
            "ppl_rev": "ppl"
        })
    elif variant == "shuffled":
        file_name = "ppl_cumulative_shuffled.csv"
        df = pd.read_csv(os.path.join(path, dataset, file_name))
        df = df.rename(columns={
            "text_id_shuff": "text_id",
            "num_sentences_shuff": "num_sentences",
            "ppl_shuff": "ppl"
        })
    elif variant == "loo":
        file_name = "ppl_leave_one_out.csv"
        df = pd.read_csv(os.path.join(path, dataset, file_name))
        # LOO: each row has (text_id, sentence_index, ppl_without_sentence)
        # We want to plot PPL at each sentence position (when that sentence is removed)
        # Rename columns to match the expected format
        df = df.rename(columns={
            "sentence_index": "num_sentences",
            "ppl_without_sentence": "ppl"
        })
        df["num_sentences"] = df["num_sentences"] + 1  # Make 1-indexed
    else:
        raise ValueError(f"Unknown variant: {variant}")
    
    df = df.sort_values(["text_id", "num_sentences"])
    return df


def compute_mean_curve(df):
    """Compute mean curve averaging across text_ids per sentence index."""
    return df.groupby("num_sentences", as_index=False)["ppl"].mean()


def plot_comparison(base_results_path: str, dataset: str, frameworks: list,
                    variants: list = None, save_path: str = None, show: bool = True,
                    debug: bool = False):
    """Plot cumulative perplexity comparing two frameworks with multiple variants."""
    if variants is None:
        variants = ["original", "reversed", "shuffled", "loo"]
    
    plt.figure(figsize=(9, 7))
    ax = plt.gca()

    if debug:
        print(f"\n{'='*60}")
        print(f"DEBUG: Dataset = {dataset}")
        print('='*60)
        
    for framework in frameworks:
        results_path = os.path.join(base_results_path, framework)
        dataset_path = os.path.join(results_path, dataset)
        
        if not os.path.exists(dataset_path):
            print(f"  Skipping {framework}/{dataset} (not found)")
            continue
        
        color = FRAMEWORK_COLORS.get(framework, "tab:gray")
        base_label = FRAMEWORK_LABELS.get(framework, framework)
        
        if debug:
            print(f"\nFramework: {framework}")
        
        for variant in variants:
            try:
                df = load_data(results_path, dataset, variant)
                mean_curve = compute_mean_curve(df)
                style = VARIANT_STYLES.get(variant, {"linestyle": "-", "label_suffix": ""})
                label = f"{base_label}{style['label_suffix']}"
                
                if debug:
                    print(f"\n  Variant: {variant}")
                    print(f"  Mean PPL per sentence:")
                    for _, row in mean_curve.iterrows():
                        print(f"    Sentence {int(row['num_sentences'])}: {row['ppl']:.2f}")
                
                # Faint per-sequence lines (only for original)
                if variant == "original":
                    for tid, g in df.groupby("text_id"):
                        plt.plot(g["num_sentences"], g["ppl"], linewidth=0.8, alpha=0.1, color=color)
                
                # Strong mean line
                plt.plot(mean_curve["num_sentences"], mean_curve["ppl"],
                         linewidth=2.5, alpha=1.0, color=color, label=label, 
                         linestyle=style["linestyle"], zorder=5)
            except Exception as e:
                print(f"  ERROR loading {framework}/{dataset}/{variant}: {e}")

    # Make y start at 0 and x ticks integers
    plt.ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(axis='both', which='major', labelsize=16)

    plt.title(f"{dataset.replace('_', ' ').title()}", fontsize=26, fontweight='bold', y=1.03)
    plt.xlabel("Sentence index", fontsize=23)
    plt.ylabel("Cumulative perplexity", fontsize=23)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend(fontsize=19, loc='upper right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def create_combined_chart(charts_path: str, output_dir: str, datasets: list, frameworks: list):
    """Create a combined chart from individual dataset plots."""
    chart_files = [os.path.join(charts_path, f'cumulative_ppl_comparison_{ds}.png') for ds in datasets]

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

    # Build title from framework labels
    framework_names = [FRAMEWORK_LABELS.get(f, f) for f in frameworks]
    # title = "Cumulative Perplexity by Sentence Ordering"
    # subtitle = " vs. ".join(framework_names)
    # fig.suptitle(title, fontsize=25, y=.96)
    # fig.text(0.5, 0.91, subtitle, ha='center', fontsize=18)
    plt.subplots_adjust(hspace=-0.2)
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, 'combined_comparison_charts.png')
    pdf_path = os.path.join(output_dir, 'combined_comparison_charts.pdf')
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"Combined chart saved as {png_path} and {pdf_path}")


def main():
    # Configuration
    BASE_RESULTS_PATH = "/home/mcedro/phd/repos/xain_xaid/XAIN_XAID/results"
    OUTPUT_DIR = "/home/mcedro/phd/repos/xain_xaid/XAIN_XAID/summary/plots/comparison"
    DATASETS = ["compas", "diabetes", "fifa", "german_credit", "stroke", "student"]
    FRAMEWORKS = [
        "templated_narrative",
        "xaistories_narratives",
        # "xaistories"
    ]
    
    # Set to True to include LOO (leave-one-out) variant
    # Note: LOO measures something different (PPL with each sentence removed)
    # and may not be directly comparable to cumulative PPL curves
    INCLUDE_LOO = False

    
    VARIANTS = ["original", "reversed", "shuffled"]
    if INCLUDE_LOO:
        VARIANTS.append("loo")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate comparison plots for each dataset
    print("Generating comparison plots per dataset...")
    for dataset in DATASETS:
        print(f"\nProcessing {dataset}...")
        save_path = os.path.join(OUTPUT_DIR, f"cumulative_ppl_comparison_{dataset}.png")
        plot_comparison(BASE_RESULTS_PATH, dataset, FRAMEWORKS, variants=VARIANTS,
                        save_path=save_path, show=False, debug=True)

    # Create combined chart of all datasets
    print("\nCreating combined chart of all datasets...")
    create_combined_chart(OUTPUT_DIR, OUTPUT_DIR, DATASETS, FRAMEWORKS)

    print("\nDone!")


if __name__ == "__main__":
    main()
