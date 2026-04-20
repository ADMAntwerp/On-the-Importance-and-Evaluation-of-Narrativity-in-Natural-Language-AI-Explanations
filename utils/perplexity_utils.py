"""Utils for calculating perplexity (ppl) using a language model"""

import json
import math
import os
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
import torch
import numpy as np
import pandas as pd
import random
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from prettytable import PrettyTable


def calculate_ppl(text, model, tokenizer, device=None):
    """
    Calculate ppl for a given text.

    Args:
        text: Input text string
        model: Language model
        tokenizer: Tokenizer
        device: Device to run on

    Returns:
        ppl value (float)
    """
    if device is None:
        device = next(model.parameters()).device

    # Tokenize
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = encodings.input_ids.to(device)

    # Calculate loss
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    # ppl is exp(loss)
    ppl = torch.exp(loss).item()

    return ppl


def reverse_sentences(text):
    """
    Reverse the order of sentences in a text.

    Args:
        text: Input text string

    Returns:
        Text with sentences in reversed order
    """
    # Split by common sentence delimiters
    import re

    # Split on .!? followed by space or end of string
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    # Reverse the order
    reversed_sentences = sentences[::-1]
    # Join back together
    return " ".join(reversed_sentences)


def shuffle_sentences(text, seed=42):
    """
    Randomly shuffle the order of sentences in a text.

    Args:
        text: Input text string
        seed: Random seed for reproducibility

    Returns:
        Text with sentences in shuffled order
    """
    import re

    # Split on .!? followed by space or end of string
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    # Set random seed for reproducibility
    random.seed(seed)

    # Shuffle the sentences
    shuffled_sentences = sentences.copy()
    random.shuffle(shuffled_sentences)

    # Join back together
    return " ".join(shuffled_sentences)


def split_sentences(text):
    """
    Split text into sentences.

    Args:
        text: Input text string

    Returns:
        List of sentences
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s.strip()]


def calculate_perplexity_change_ratio(ppl_original, ppl_shuffled):
    """
    Calculate the ratio of change in perplexity when sentences are shuffled.

    Formula: (ppl_shuffled - ppl_original) / ppl_original

    Args:
        ppl_original (float): Perplexity of the original text
        ppl_shuffled (float): Perplexity of the shuffled text

    Returns:
        float: The ratio (ppl_shuffled - ppl_original) / ppl_original

    Notes:
        - Positive ratio: Shuffled text has higher perplexity (less predictable)
        - Negative ratio: Shuffled text has lower perplexity (more predictable)
        - Returns None if ppl_original is 0 or None
    """
    if ppl_original is None or ppl_original == 0:
        return None

    if ppl_shuffled is None:
        return None

    ratio = (ppl_shuffled - ppl_original) / ppl_original
    return ratio


def calculate_perplexity_change_ratio_reversed(ppl_original, ppl_reversed):
    """
    Calculate the ratio of change in perplexity when sentences are reversed.

    Formula: (ppl_reversed - ppl_original) / ppl_original

    Args:
        ppl_original (float): Perplexity of the original text
        ppl_reversed (float): Perplexity of the reversed text

    Returns:
        float: The ratio (ppl_reversed - ppl_original) / ppl_original

    Notes:
        - Positive ratio: Reversed text has higher perplexity (less predictable)
        - Negative ratio: Reversed text has lower perplexity (more predictable)
        - Returns None if ppl_original is 0 or None
    """
    if ppl_original is None or ppl_original == 0:
        return None

    if ppl_reversed is None:
        return None

    ratio = (ppl_reversed - ppl_original) / ppl_original
    return ratio


def calculate_perplexity_change_ratio_loo(ppl_original, ppl_loo_mean):
    """
    Calculate the ratio of change in perplexity using leave-one-out mean.

    Formula: (ppl_loo_mean - ppl_original) / ppl_original

    Args:
        ppl_original (float): Perplexity of the original text
        ppl_loo_mean (float): Mean perplexity from leave-one-out analysis

    Returns:
        float: The ratio (ppl_loo_mean - ppl_original) / ppl_original

    Notes:
        - Positive ratio: Leave-one-out mean has higher perplexity (removing sentences increases confusion)
        - Negative ratio: Leave-one-out mean has lower perplexity (removing sentences decreases confusion)
        - Returns None if ppl_original is 0 or None
    """
    if ppl_original is None or ppl_original == 0:
        return None

    if ppl_loo_mean is None:
        return None

    ratio = (ppl_loo_mean - ppl_original) / ppl_original
    return ratio


def calculate_ppl_leave_one_out(text, model, tokenizer):
    """
    Calculate ppl for text with each sentence removed one at a time.

    Args:
        text: Input text string
        model: Language model
        tokenizer: Tokenizer

    Returns:
        List of dictionaries with sentence index, removed sentence, and ppl
    """
    sentences = split_sentences(text)

    if len(sentences) <= 1:
        # Can't do leave-one-out with 1 or fewer sentences
        print("[INFO] Not enough sentences for leave-one-out.")
        return []

    results = []

    for i in range(len(sentences)):
        # Create text without sentence i
        remaining_sentences = sentences[:i] + sentences[i + 1 :]
        remaining_text = " ".join(remaining_sentences)

        if remaining_text.strip():
            ppl = calculate_ppl(remaining_text, model, tokenizer)
            results.append(
                {
                    "sentence_index": i,
                    "removed_sentence": sentences[i],
                    "ppl": ppl,
                }
            )

    return results


def load_perplexity_data(json_path):
    """Load perplexity data from JSON file"""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def calculate_ppl_cumulative(text, model, tokenizer):
    """
    Calculate ppl for cumulative sentences (1, 1-2, 1-3, etc.).

    Args:
        text: Input text string
        model: Language model
        tokenizer: Tokenizer

    Returns:
        List of dictionaries with sentence count, cumulative text, and ppl
    """
    sentences = split_sentences(text)

    if len(sentences) == 0:
        return []

    results = []

    for i in range(1, len(sentences) + 1):
        # Take sentences from 0 to i (cumulative)
        cumulative_sentences = sentences[:i]
        cumulative_text = " ".join(cumulative_sentences)

        if cumulative_text.strip():
            ppl = calculate_ppl(cumulative_text, model, tokenizer)
            results.append(
                {
                    "num_sentences": i,
                    "last_sentence": sentences[i - 1],
                    "cumulative_text": cumulative_text,
                    "ppl": ppl,
                }
            )

    return results


def calculate_ppl_cumulative_reversed(text, model, tokenizer):
    """
    Calculate ppl for cumulative sentences (1, 1-2, 1-3, etc.).

    Args:
        text: Input text string
        model: Language model
        tokenizer: Tokenizer

    Returns:
        List of dictionaries with sentence count, cumulative text, and ppl
    """
    reversed = reverse_sentences(text)
    reversed_text = split_sentences(reversed)

    if len(reversed_text) == 0:
        return []

    results = []

    for i in range(1, len(reversed_text) + 1):
        # Take sentences from 0 to i (cumulative)
        cumulative_sentences = reversed_text[:i]
        cumulative_text = " ".join(cumulative_sentences)

        if cumulative_text.strip():
            ppl = calculate_ppl(cumulative_text, model, tokenizer)
            results.append(
                {
                    "num_sentences": i,
                    "last_sentence": reversed_text[i - 1],
                    "cumulative_text": cumulative_text,
                    "ppl": ppl,
                }
            )

    return results


def calculate_ppl_cumulative_shuffled(text, model, tokenizer, seed=42):
    """
    Calculate ppl for cumulative sentences after shuffling (1, 1-2, 1-3, etc.).

    Args:
        text: Input text string
        model: Language model
        tokenizer: Tokenizer
        seed: Random seed for shuffling

    Returns:
        List of dictionaries with sentence count, cumulative text, and ppl
    """
    sentences = split_sentences(text)

    if len(sentences) == 0:
        return []

    # Shuffle sentences with seed
    random.seed(seed)
    shuffled_sentences = sentences.copy()
    random.shuffle(shuffled_sentences)

    results = []

    for i in range(1, len(shuffled_sentences) + 1):
        # Take sentences from 0 to i (cumulative)
        cumulative_sentences = shuffled_sentences[:i]
        cumulative_text = " ".join(cumulative_sentences)

        if cumulative_text.strip():
            ppl = calculate_ppl(cumulative_text, model, tokenizer)
            results.append(
                {
                    "num_sentences": i,
                    "last_sentence": shuffled_sentences[i - 1],
                    "cumulative_text": cumulative_text,
                    "ppl": ppl,
                }
            )

    return results


def create_summary_table(results_folder, *, pretty=False, title="Summary"):
    """
    Creates a summary table aggregating results from different metrics.
    """
    summary_data = []
    methods = sorted(
        d
        for d in os.listdir(results_folder)
        if os.path.isdir(os.path.join(results_folder, d))
    )

    for method in methods:
        method_path = os.path.join(results_folder, method)

        datasets = sorted(
            d
            for d in os.listdir(method_path)
            if os.path.isdir(os.path.join(method_path, d))
        )

        for dataset in datasets:
            dataset_path = os.path.join(method_path, dataset)
            row = {"Method": method, "Dataset": dataset}

            # connectives_ratio_results.csv
            conn_path = os.path.join(dataset_path, "connectives_ratios_results.csv")
            if os.path.exists(conn_path):
                df_conn = pd.read_csv(conn_path)
                row["Conn_ratio"] = round(df_conn["connectives_ratio"].mean(), 2)
                row["Conn_ratio_std"] = round(df_conn["connectives_ratio"].std(), 2)

            # ppl_statistics.csv
            ppl_path = os.path.join(dataset_path, "ppl_statistics.csv")
            if os.path.exists(ppl_path):
                df_ppl = pd.read_csv(ppl_path)
                row["PPL_mean_ori"] = round(df_ppl["mean_ori"].mean(), 2)
                row["PPL_std_ori"] = round(df_ppl["std_ori"].mean(), 2)

            # ttr_results.csv
            ttr_path = os.path.join(dataset_path, "ttr_results.csv")
            if os.path.exists(ttr_path):
                df_ttr = pd.read_csv(ttr_path)
                row["TTR"] = round(df_ttr["TTR"].mean(), 2)
                row["TTR_std"] = round(df_ttr["TTR"].std(), 2)

            # verb_ratios_results.csv
            verb_path = os.path.join(dataset_path, "verb_ratios_results.csv")
            if os.path.exists(verb_path):
                df_verb = pd.read_csv(verb_path)
                row["Verb_ratio"] = round(df_verb["verb_ratio"].mean(), 2)
                row["Verb_ratio_std"] = round(df_verb["verb_ratio"].std(), 2)

            summary_data.append(row)

    # DataFrame
    summary_df = pd.DataFrame(summary_data)

    cols = [
        "Method",
        "Dataset",
        "PPL_mean_ori",
        "PPL_std_ori",
        "TTR",
        "TTR_std",
        "Verb_ratio",
        "Verb_ratio_std",
        "Conn_ratio",
        "Conn_ratio_std",
    ]
    present_cols = [c for c in cols if c in summary_df.columns]
    summary_df = summary_df.reindex(columns=present_cols)

    if not pretty:
        return summary_df

    # Convert to PrettyTable
    table = PrettyTable()
    table.title = title
    table.field_names = present_cols
    table.add_rows(summary_df.to_records(index=False).tolist())
    table.align = "l"  # left-align all columns
    table.max_width = 28  # optional: wrap long cells to keep the table narrow

    return table


def create_xaistories_table(data):
    """
    Create a detailed table for xaistories with all available metrics.
    """
    rows = []

    if "llm_outputs" in data:
        for key, values in data["llm_outputs"].items():
            method, dataset = key.split("/")
            if method != "xaistories":
                continue

            row = {
                "Dataset": dataset,
                "mean_ori": values.get("mean_ori", "NA"),
                "median_ori": values.get("median_ori", "NA"),
                "std_ori": values.get("std_ori", "NA"),
                "min_ori": values.get("min_ori", "NA"),
                "max_ori": values.get("max_ori", "NA"),
                "mean_reversed": values.get("mean_reversed", "NA"),
                "median_reversed": values.get("median_reversed", "NA"),
                "std_reversed": values.get("std_reversed", "NA"),
                "min_reversed": values.get("min_reversed", "NA"),
                "max_reversed": values.get("max_reversed", "NA"),
                "mean_shuffled": values.get("mean_shuffled", "NA"),
                "median_shuffled": values.get("median_shuffled", "NA"),
                "std_shuffled": values.get("std_shuffled", "NA"),
                "min_shuffled": values.get("min_shuffled", "NA"),
                "max_shuffled": values.get("max_shuffled", "NA"),
                "loo_mean_of_means": values.get("loo_mean_of_means", "NA"),
                "loo_mean_of_stds": values.get("loo_mean_of_stds", "NA"),
                "loo_mean_of_mins": values.get("loo_mean_of_mins", "NA"),
                "loo_mean_of_maxs": values.get("loo_mean_of_maxs", "NA"),
                "loo_overall_std": values.get("loo_overall_std", "NA"),
                "cumulative_mean_slope": values.get("cumulative_mean_slope", "NA"),
                "cumulative_std_slope": values.get("cumulative_std_slope", "NA"),
                "cumulative_rev_mean_slope": values.get(
                    "cumulative_rev_mean_slope", "NA"
                ),
                "cumulative_rev_std_slope": values.get(
                    "cumulative_rev_std_slope", "NA"
                ),
                "cumulative_shuff_mean_slope": values.get(
                    "cumulative_shuff_mean_slope", "NA"
                ),
                "cumulative_shuff_std_slope": values.get(
                    "cumulative_shuff_std_slope", "NA"
                ),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by Dataset
    df = df.sort_values("Dataset").reset_index(drop=True)

    return df


def df_to_pretty(df: pd.DataFrame, title: str | None = None) -> PrettyTable:
    t = PrettyTable()
    if title:
        t.title = title
    t.field_names = list(df.columns)
    t.add_rows(df.to_records(index=False).tolist())
    t.align = "l"
    t.max_width = 28  # optional
    return t


def get_dataset_names(folder_path):
    """Extract dataset names from CSV files in the folder."""
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist, using default datasets")
        return ["compas", "diabetes", "fifa", "german_credit", "stroke", "student"]

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    # Remove .csv extension to get dataset names
    dataset_names = [os.path.splitext(f)[0] for f in csv_files]
    return sorted(dataset_names)


def plot_cumulative_perplexity(
    path, save_dir, dataset, method="xaistories", save_fig=True
):
    """
    Generate cumulative perplexity plot for a given dataset.

    Args:
        dataset: Dataset name
        path: Base path to results
        save_fig: Whether to save the figure to file
    """
    print(f"Processing dataset: xaistories/{dataset}")

    ori_path = os.path.join(path, method, dataset, "ppl_cumulative.csv")
    shuf_path = os.path.join(path, method, dataset, "ppl_cumulative_shuffled.csv")
    rev_path = os.path.join(path, method, dataset, "ppl_cumulative_rev.csv")

    # Check if files exist
    if not all(os.path.exists(p) for p in [ori_path, shuf_path, rev_path]):
        print(f"[INFO] Warning: Missing files for {dataset}, skipping...")
        return

    df_ori = pd.read_csv(ori_path)
    df_shuf = pd.read_csv(shuf_path)
    df_rev = pd.read_csv(rev_path)

    df_shuf = df_shuf.rename(
        columns={
            "text_id_shuff": "text_id",
            "num_sentences_shuff": "num_sentences",
            "last_sentence_shuff": "last_sentence",
            "ppl_shuff": "ppl",
        }
    )
    df_rev = df_rev.rename(
        columns={
            "text_id_rev": "text_id",
            "num_sentences_rev": "num_sentences",
            "last_sentence_rev": "last_sentence",
            "ppl_rev": "ppl",
        }
    )

    # Sort by text_id and num_sentences
    df_ori = df_ori.sort_values(["text_id", "num_sentences"])
    df_shuf = df_shuf.sort_values(["text_id", "num_sentences"])
    df_rev = df_rev.sort_values(["text_id", "num_sentences"])

    # Compute mean curves
    mean_ori = df_ori.groupby("num_sentences", as_index=False)["ppl"].mean()
    mean_shuf = df_shuf.groupby("num_sentences", as_index=False)["ppl"].mean()
    mean_rev = df_rev.groupby("num_sentences", as_index=False)["ppl"].mean()

    # Plot
    plt.figure(figsize=(8, 7))
    ax = plt.gca()

    # Faint per-sequence lines
    for tid, g in df_ori.groupby("text_id"):
        plt.plot(
            g["num_sentences"], g["ppl"], linewidth=1.0, alpha=0.25, color="tab:blue"
        )
    for tid, g in df_shuf.groupby("text_id"):
        plt.plot(
            g["num_sentences"], g["ppl"], linewidth=1.0, alpha=0.25, color="tab:orange"
        )
    for tid, g in df_rev.groupby("text_id"):
        plt.plot(
            g["num_sentences"], g["ppl"], linewidth=1.0, alpha=0.25, color="tab:green"
        )

    # Strong mean lines
    plt.plot(
        mean_ori["num_sentences"],
        mean_ori["ppl"],
        linewidth=3.0,
        alpha=1.0,
        color="tab:blue",
        label="Original (mean)",
        zorder=5,
    )
    plt.plot(
        mean_shuf["num_sentences"],
        mean_shuf["ppl"],
        linewidth=3.0,
        alpha=1.0,
        color="tab:orange",
        label="Shuffled (mean)",
        zorder=5,
    )
    plt.plot(
        mean_rev["num_sentences"],
        mean_rev["ppl"],
        linewidth=3.0,
        alpha=1.0,
        color="tab:green",
        label="Reversed (mean)",
        zorder=5,
    )

    # Format plot
    plt.ylim(bottom=0)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.title(f"Cumulative Perplexity by Sentence: {dataset}")
    plt.xlabel("Sentence index")
    plt.ylabel("Cumulative perplexity")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(save_dir, "plots")
    os.makedirs(output_path, exist_ok=True)
    if save_fig:
        output_file = f"{output_path}/cumulative_ppl_{dataset}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  Saved: {output_file}\n")

    plt.close()


def calculate_distinct_bigram_ratio(text):
    """
    Calculate the distinct bigram ratio of a text.

    Args:
        text: Input text string

    Returns:
        float: Ratio of distinct bigrams to total bigrams
    """
    # Split text into words (lowercased)
    words = text.lower().split()

    if len(words) < 2:
        return 0.0

    # Create bigrams
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]

    # Count distinct bigrams
    distinct_bigrams = len(set(bigrams))
    total_bigrams = len(bigrams)

    # Calculate ratio
    ratio = distinct_bigrams / total_bigrams if total_bigrams > 0 else 0.0

    return ratio


def calculate_fdr(text, perplexity):
    """Calculate the FDR score.

    Formula:
    fdr = (distinct 2-gram ratio)^2 / ln(perplexity)
    """
    if math.isnan(perplexity) or perplexity <= 0 or perplexity == float("inf"):
        return 0.0

    distinct_bigram_ratio = calculate_distinct_bigram_ratio(text)
    ln_perplexity = math.log(perplexity)

    if ln_perplexity == 0:
        return 0.0

    fdr = (distinct_bigram_ratio ** 2) / ln_perplexity

    return fdr



def compute_fdr_for_json(json_path, model, tokenizer, output_dir):
    """Compute FDR for each text in a JSON file.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    fdr_scores = {}
    fdr_values = []
    bigram_ratios = []
    perplexities = []
    results_list = []

    os.makedirs(output_dir, exist_ok=True)

    for key, text in tqdm(
        data.items(), desc=f"Computing FDR for {Path(json_path).name}"
    ):
        if isinstance(text, str) and text.strip():
            # Calculate perplexity
            ppl = calculate_ppl(text, model, tokenizer)
            perplexities.append(ppl)

            # Calculate distinct bigram ratio
            bigram_ratio = calculate_distinct_bigram_ratio(text)
            bigram_ratios.append(bigram_ratio)

            # Calculate FDR (same formula as legacy CEM)
            fdr = calculate_fdr(text, ppl)
            fdr_scores[key] = fdr
            fdr_values.append(fdr)

            # Store detailed results (include legacy cem key for compatibility)
            results_list.append(
                {
                    "key": key,
                    "fdr": fdr,
                    "cem": fdr,  # Backward-compatible alias  
                    "distinct_bigram_ratio": bigram_ratio,
                    "perplexity": ppl,
                    "ln_perplexity": math.log(ppl) if ppl > 0 else 0,
                }
            )

    # Calculate statistics
    fdr_stats = {
        "mean": float(np.mean(fdr_values)) if fdr_values else 0.0,
        "std": float(np.std(fdr_values)) if fdr_values else 0.0,
        "min": float(np.min(fdr_values)) if fdr_values else 0.0,
        "max": float(np.max(fdr_values)) if fdr_values else 0.0,
        "median": float(np.median(fdr_values)) if fdr_values else 0.0,
    }

    statistics = {
        "fdr": fdr_stats,
        "cem": fdr_stats,
        "distinct_bigram_ratio": {
            "mean": float(np.mean(bigram_ratios)) if bigram_ratios else 0.0,
            "std": float(np.std(bigram_ratios)) if bigram_ratios else 0.0,
            "min": float(np.min(bigram_ratios)) if bigram_ratios else 0.0,
            "max": float(np.max(bigram_ratios)) if bigram_ratios else 0.0,
            "median": float(np.median(bigram_ratios)) if bigram_ratios else 0.0,
        },
        "perplexity": {
            "mean": float(np.mean(perplexities)) if perplexities else 0.0,
            "std": float(np.std(perplexities)) if perplexities else 0.0,
            "min": float(np.min(perplexities)) if perplexities else 0.0,
            "max": float(np.max(perplexities)) if perplexities else 0.0,
            "median": float(np.median(perplexities)) if perplexities else 0.0,
        },
    }

    # Save detailed results to CSV (new and legacy filenames)
    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(output_dir, "fdc_detailed.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "cem_detailed.csv"), index=False)

    return {
        "fdr_scores": fdr_scores,
        "cem_scores": fdr_scores,
        "detailed_results": results_list,
        "statistics": statistics,
    }


def compute_cem_for_json(json_path, model, tokenizer, output_dir):
    """
    Compute CEM (Cause-Effect Metric) for each text in a JSON file.

    Args:
        json_path: Path to JSON file containing texts
        model: Language model for perplexity calculation
        tokenizer: Tokenizer for the model
        output_dir: Directory to save results

    Returns:
        Dictionary with CEM results and statistics
    """
    # Backward-compatible alias; logic is implemented in compute_fdr_for_json.
    return compute_fdr_for_json(json_path, model, tokenizer, output_dir)


def compute_cumulative_ppl_and_dcpr(json_path, model, tokenizer, output_dir, method_name, dataset_name):
    """
    Compute cumulative perplexity curves and fit shifted exponential model:
    y(x) = A * r^x + C
    
    (Internally fits y = A * exp(-bx) + C, then converts b -> r)

    Strategy:
    - Uses "Dynamic Bounding" to constrain C (asymptote) to be physically
      realistic based on the dataset's minimum observed perplexity.
    - Handles short datasets (len <= 3) by tightening bounds to prevent overfitting.
    """
    from scipy.optimize import curve_fit

    with open(json_path, "r") as f:
        data = json.load(f)

    # Collect cumulative perplexities and diversity metrics
    cumulative_ppls_by_length = {} 
    distinct_2gram_ratios = []
    ttr_values = []
    vr_values = []
    cd_values = []
    detailed_results = []

    os.makedirs(output_dir, exist_ok=True)

    print(f"Computing cumulative perplexities for {Path(json_path).name}...")

    for key, text in tqdm(data.items(), desc="Processing texts"):
        if isinstance(text, str) and text.strip():
            sentences = split_sentences(text)

            if len(sentences) < 2:
                continue

            cumulative_text = ""
            cumulative_ppls = []

            for i, sentence in enumerate(sentences):
                if i == 0:
                    cumulative_text = sentence
                else:
                    cumulative_text += " " + sentence

                ppl = calculate_ppl(cumulative_text, model, tokenizer)
                cumulative_ppls.append(ppl)

                position = i + 1 
                if position not in cumulative_ppls_by_length:
                    cumulative_ppls_by_length[position] = []
                cumulative_ppls_by_length[position].append(ppl)

            distinct_2gram_ratio = calculate_distinct_bigram_ratio(text)
            distinct_2gram_ratios.append(distinct_2gram_ratio)

            # Compute additional diversity metrics
            from get_metrics.lexical_diversity_ttr import compute_ttr
            from get_metrics.lexical_diversity_vr import compute_verb_ratio
            from get_metrics.connectives_all import compute_connectives_ratio
            from utils.connectives_lex import connectives

            ttr_val = compute_ttr(text)
            vr_val = compute_verb_ratio(text)
            cd_val = compute_connectives_ratio(text, connectives)
            ttr_values.append(ttr_val)
            vr_values.append(vr_val)
            cd_values.append(cd_val)

            detailed_results.append({
                "text_id": key,
                "num_sentences": len(sentences),
                "distinct_2gram_ratio": round(distinct_2gram_ratio, 6),
                "ttr": round(ttr_val, 6),
                "verb_ratio": round(vr_val, 6),
                "connectives_ratio": round(cd_val, 6),
                "cumulative_ppls": [round(p, 4) for p in cumulative_ppls],
            })

    if not cumulative_ppls_by_length:
        return {"error": "No valid texts found"}

    # Average perplexities
    x_values = []
    y_values = []

    for position in sorted(cumulative_ppls_by_length.keys()):
        ppls = cumulative_ppls_by_length[position]
        avg_ppl = np.mean(ppls)
        x_values.append(position)
        y_values.append(avg_ppl)

    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # --- NEW FITTING LOGIC ---

    # Define model: y = A * exp(-bx) + C
    def exp_offset_func(x, a, b, c):
        return a * np.exp(-b * x) + c

    # Determine Dynamic Bounds to solve "Tail Drag"
    min_ppl = np.min(y_values)
    max_ppl = np.max(y_values)
    
    # Heuristic: If dataset is very short (<= 3 points), we can't reliably find C.
    # We constrain C to be slightly below the minimum to stabilize the curve.
    is_short_dataset = len(x_values) <= 3
    
    try:
        if is_short_dataset:
            # Fallback for short data: Fix C roughly 1.0 unit below min
            estimated_c = max(0.1, min_ppl - 1.0)
            c_lower = estimated_c - 0.05
            c_upper = estimated_c + 0.05
        else:
            # Standard for full data: 
            # C must be > 0.1 and strictly lower than the data curve (min_ppl)
            c_lower = 0.1
            c_upper = min_ppl - 0.01 

        # Bounds: [A_min, b_min, C_min], [A_max, b_max, C_max]
        # b is decay constant (must be > 0)
        bounds = (
            [0.1,     0.001, c_lower], 
            [np.inf,  10.0,  c_upper]
        )

        # Initial Guess
        p0 = [max_ppl - min_ppl, 0.5, min_ppl - 1.0]

        popt, pcov = curve_fit(
            exp_offset_func,
            x_values,
            y_values,
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        
        A_opt, b_opt, C_opt = popt
        
        # CRITICAL: Convert decay constant b to geometric rate r
        r_val = np.exp(-b_opt)
        fit_success = True

    except Exception as e:
        print(f"Warning: Curve fitting failed: {e}")
        A_opt, b_opt, C_opt, r_val = None, None, None, None
        fit_success = False

    # Calculate average distinct 2-gram ratio
    avg_distinct_2gram_ratio = (
        np.mean(distinct_2gram_ratios) if distinct_2gram_ratios else 0
    )

    # Calculate average TTR, VR, CD
    avg_ttr = np.mean(ttr_values) if ttr_values else 0
    avg_vr = np.mean(vr_values) if vr_values else 0
    avg_cd = np.mean(cd_values) if cd_values else 0

    # Calculate WAR metrics using r (not b)
    if fit_success and r_val is not None and avg_distinct_2gram_ratio > 0:
        dcpr = r_val / (avg_distinct_2gram_ratio ** 2)
    else:
        dcpr = None

    if fit_success and r_val is not None and avg_cd > 0:
        ccpr = r_val / (avg_cd ** 2)
    else:
        ccpr = None

    if fit_success and r_val is not None and avg_ttr > 0:
        ttcpr = r_val / (avg_ttr ** 2)
    else:
        ttcpr = None

    if fit_success and r_val is not None and avg_vr > 0:
        vcpr = r_val / (avg_vr ** 2)
    else:
        vcpr = None

    # Calculate goodness-of-fit metrics
    fit_quality = {}
    if fit_success:
        # Fitted values
        y_fitted = exp_offset_func(x_values, A_opt, b_opt, C_opt)

        # Residuals
        residuals = y_values - y_fitted
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean(residuals**2))
        mape = np.mean(np.abs(residuals / y_values)) * 100 if np.all(y_values != 0) else 0

        fit_quality = {
            "r_squared": round(r_squared, 6),
            "rmse": round(rmse, 6),
            "mape": round(mape, 6),
            "residual_mean": round(np.mean(residuals), 6),
            "residual_std": round(np.std(residuals), 6),
        }

    # Prepare results
    results = {
        "fitted_parameters": {
            "A": round(A_opt, 6) if fit_success else None,
            "r": round(r_val, 6) if fit_success else None,
            "C": round(C_opt, 6) if fit_success else None,
            "b_decay": round(b_opt, 6) if fit_success else None,
            "fit_success": fit_success,
        },
        "fit_quality": fit_quality,
        "dataset_statistics": {
            "num_texts": len(detailed_results),
            "num_sentences_range": [
                min(r["num_sentences"] for r in detailed_results),
                max(r["num_sentences"] for r in detailed_results),
            ],
            "avg_distinct_2gram_ratio": round(avg_distinct_2gram_ratio, 6),
            "avg_ttr": round(avg_ttr, 6),
            "avg_verb_ratio": round(avg_vr, 6),
            "avg_connectives_ratio": round(avg_cd, 6),
        },
        "dcpr_metric": round(dcpr, 3) if dcpr is not None else None,
        "ccpr_metric": round(ccpr, 3) if ccpr is not None else None,
        "ttcpr_metric": round(ttcpr, 3) if ttcpr is not None else None,
        "vcpr_metric": round(vcpr, 3) if vcpr is not None else None,
        "averaged_curve": {
            "x": x_values.tolist(),
            "y": [round(y, 4) for y in y_values],
        },
    }

    # Save detailed results
    detailed_df = pd.DataFrame(detailed_results)
    detailed_df.to_csv(os.path.join(output_dir, "cumulative_ppl_detailed.csv"), index=False)

    # Save summary results
    with open(os.path.join(output_dir, "cumulative_ppl_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved analysis results to: {output_dir}")

    # --- PLOTTING ---
    if fit_success:
        x_plot = np.linspace(1, max(x_values), 100)
        y_plot = exp_offset_func(x_plot, A_opt, b_opt, C_opt)

        r_squared = fit_quality.get("r_squared", 0)
        rmse = fit_quality.get("rmse", 0)
        mape = fit_quality.get("mape", 0)

        # Main plot
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.scatter(x_values, y_values, label="Averaged data", s=120, alpha=0.8, color="steelblue", edgecolors="black")
        
        # Label matches paper: A * r^x + C
        label_text = f"Fit: $y = {A_opt:.2f} \\cdot {r_val:.3f}^x + {C_opt:.2f}$"
        ax.plot(x_plot, y_plot, "r-", label=label_text, linewidth=2.5)
        
        ax.set_xlabel("Number of Sentences (Cumulative)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Perplexity", fontsize=13, fontweight="bold")

        title_text = f"Cumulative Perplexity (Offset Exponential) of {method_name.title()}: {dataset_name.title()}\n"
        title_text += f"$A={A_opt:.2f}$ | $r={r_val:.4f}$ | $C={C_opt:.2f}$ | $dcpr={dcpr:.4f}$\n"
        title_text += f"$ccpr={ccpr:.4f}$ | $ttcpr={ttcpr:.4f}$ | $vcpr={vcpr:.4f}$\n" if ccpr is not None and ttcpr is not None and vcpr is not None else ""
        title_text += f"$R^2={r_squared:.4f}$ | RMSE={rmse:.4f} | MAPE={mape:.2f}%"
        
        ax.set_title(title_text, fontsize=12, fontweight="bold")
        ax.legend(fontsize=12, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, "cumulative_ppl_fit.png"), dpi=300, bbox_inches="tight")
        print(f"Saved fit plot.")
        plt.close()

        # Residuals plot
        residuals = y_values - exp_offset_func(x_values, A_opt, b_opt, C_opt)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.scatter(x_values, residuals, s=120, alpha=0.8, color="darkred", edgecolors="black")
        ax.axhline(y=0, color="black", linestyle="--")
        ax.set_xlabel("Number of Sentences", fontsize=13)
        ax.set_ylabel("Residuals", fontsize=13)
        ax.set_title(f"Fit Residuals (Mean: {np.mean(residuals):.4f})", fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cumulative_ppl_residuals.png"), dpi=300)
        plt.close()

    return results


def plot_cumulative_ppl_comparison(all_results_dict, output_dir):
    """
    Create comparison plots across all datasets/methods.

    Args:
        all_results_dict: Dictionary with structure {method/dataset: results}
        output_dir: Directory to save comparison plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data for plotting
    plot_data = {}
    dcpr_values = {}

    for identifier, results in all_results_dict.items():
        if "error" not in results and results["fitted_parameters"]["fit_success"]:
            curve = results["averaged_curve"]
            x_vals = np.array(curve["x"])
            y_vals = np.array(curve["y"])
            A = results["fitted_parameters"]["A"]
            r = results["fitted_parameters"]["r"]
            dcpr = results["dcpr_metric"]

            plot_data[identifier] = {
                "x": x_vals,
                "y": y_vals,
                "A": A,
                "r": r,
            }
            dcpr_values[identifier] = dcpr

    if not plot_data:
        print("No successful fits found for comparison plotting")
        return

    # Plot 1: All curves overlaid
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(plot_data)))

    for (identifier, data), color in zip(plot_data.items(), colors):
        x_vals = data["x"]
        y_vals = data["y"]
        A = data["A"]
        r = data["r"]
        dcpr = dcpr_values[identifier]

        # Plot data points
        ax.scatter(
            x_vals,
            y_vals,
            s=80,
            alpha=0.6,
            color=color,
            edgecolors="black",
            linewidth=0.5,
        )

        # Plot fitted curve
        x_plot = np.linspace(x_vals.min(), x_vals.max(), 50)
        y_plot = A * np.power(r, x_plot)
        ax.plot(
            x_plot,
            y_plot,
            "-",
            color=color,
            linewidth=2.5,
            label=f"{identifier} ($\\dcpr$={dcpr:.4f})",
        )

    ax.set_xlabel("Number of Sentences (Cumulative)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Perplexity", fontsize=13, fontweight="bold")
    ax.set_title(
        "Cumulative Perplexity Growth - All Datasets Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="best", ncol=2)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()

    comparison_path = os.path.join(output_dir, "cumulative_ppl_all_comparison.png")
    plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
    print(f"Saved comparison plot to: {comparison_path}")
    plt.close()

    # Plot 2: dcpr metric comparison
    if dcpr_values:
        fig, ax = plt.subplots(figsize=(12, 6))
        identifiers = list(dcpr_values.keys())
        dcprs = list(dcpr_values.values())
        colors_bar = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(dcprs)))

        bars = ax.bar(
            range(len(identifiers)),
            dcprs,
            color=colors_bar,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on bars
        for i, (bar, dcpr) in enumerate(zip(bars, dcpr_values.values())):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{dcpr:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xticks(range(len(identifiers)))
        ax.set_xticklabels(identifiers, rotation=45, ha="right")
        ax.set_ylabel("dcpr Metric", fontsize=13, fontweight="bold")
        ax.set_title(
            "dcpr Metric Comparison - Lower is Better (Higher Continuity)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.set_axisbelow(True)
        plt.tight_layout()

        dcpr_plot_path = os.path.join(output_dir, "dcpr_metric_comparison.png")
        plt.savefig(dcpr_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved dcpr comparison plot to: {dcpr_plot_path}")
        plt.close()

    # Plot 3: R-value (growth rate) comparison
    r_values = {identifier: data["r"] for identifier, data in plot_data.items()}
    if r_values:
        fig, ax = plt.subplots(figsize=(12, 6))
        identifiers = list(r_values.keys())
        rs = list(r_values.values())
        colors_bar = plt.cm.coolwarm(np.linspace(0, 1, len(rs)))

        bars = ax.bar(
            range(len(identifiers)),
            rs,
            color=colors_bar,
            edgecolor="black",
            linewidth=1.5,
        )

        # Add value labels on bars
        for i, (bar, r) in enumerate(zip(bars, rs)):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{r:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xticks(range(len(identifiers)))
        ax.set_xticklabels(identifiers, rotation=45, ha="right")
        ax.set_ylabel("Growth Rate (r)", fontsize=13, fontweight="bold")
        ax.set_title(
            "Exponential Growth Rate Comparison - Lower is Better",
            fontsize=14,
            fontweight="bold",
        )
        ax.axhline(
            y=1.0, color="red", linestyle="--", linewidth=2, label="r=1 (no growth)"
        )
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.legend(fontsize=11)
        ax.set_axisbelow(True)
        plt.tight_layout()

        r_plot_path = os.path.join(output_dir, "growth_rate_comparison.png")
        plt.savefig(r_plot_path, dpi=300, bbox_inches="tight")
        print(f"Saved growth rate comparison plot to: {r_plot_path}")
        plt.close()


def process_json_file(json_path, model, tokenizer, output_dir, method, shuffle_seed=42):
    """
    Process a JSON file and calculate ppl for each text.

    Args:
        json_path: Path to JSON file
        model: Language model
        tokenizer: Tokenizer
        output_dir: Directory to save CSV results
        method: Method name (if "xaistories", calculate full metrics; otherwise basic only)
        shuffle_seed: Random seed for sentence shuffling

    Returns:
        Dictionary with ppl values and statistics
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    is_full_analysis = method.lower() in ["templated_narrative", "explingo_zero_shot", "explingo_narratives", "xaistories", "xaistories_narratives"]

    perplexities = {}
    ppl_values = []
    results_list = []

    # Only initialize these for full analysis
    if is_full_analysis:
        perplexities_reversed = {}
        perplexities_shuffled = {}
        ppl_reversed_values = []
        ppl_shuffled_values = []
        leave_one_out_all = []
        cumulative_all = []
        cumulative_rev_all = []
        cumulative_shuff_all = []

        all_loo_means = []
        all_loo_stds = []
        all_loo_mins = []
        all_loo_maxs = []

        all_cumulative_slopes = []
        all_cumulative_rev_slopes = []
        all_cumulative_shuff_slopes = []

    for key, text in tqdm(data.items(), desc=f"Processing {Path(json_path).name}"):
        if isinstance(text, str) and text.strip():
            # Always calculate ppl for original text
            ppl = calculate_ppl(text, model, tokenizer)
            perplexities[key] = ppl
            ppl_values.append(ppl)

            if is_full_analysis:
                # Calculate ppl for reversed text
                reversed_text = reverse_sentences(text)
                ppl_reversed = calculate_ppl(reversed_text, model, tokenizer)
                perplexities_reversed[key] = ppl_reversed
                ppl_reversed_values.append(ppl_reversed)

                # Calculate ppl for shuffled text
                shuffled_text = shuffle_sentences(text, seed=shuffle_seed)
                ppl_shuffled = calculate_ppl(shuffled_text, model, tokenizer)
                perplexities_shuffled[key] = ppl_shuffled
                ppl_shuffled_values.append(ppl_shuffled)

                # Calculate leave-one-out perplexities
                loo_results = calculate_ppl_leave_one_out(text, model, tokenizer)

                # Calculate cumulative perplexities
                cumulative_results = calculate_ppl_cumulative(text, model, tokenizer)

                # Calculate reversed cumulative perplexities
                cumulative_rev_results = calculate_ppl_cumulative_reversed(
                    text, model, tokenizer
                )

                # Calculate shuffled cumulative perplexities
                cumulative_shuff_results = calculate_ppl_cumulative_shuffled(
                    text, model, tokenizer, seed=shuffle_seed
                )

                # Calculate statistics for leave-one-out
                if loo_results:
                    loo_perplexities = [r["ppl"] for r in loo_results]
                    loo_mean = np.mean(loo_perplexities)
                    loo_std = np.std(loo_perplexities)
                    loo_min = np.min(loo_perplexities)
                    loo_max = np.max(loo_perplexities)
                    num_sentences = len(loo_results)

                    all_loo_means.append(loo_mean)
                    all_loo_stds.append(loo_std)
                    all_loo_mins.append(loo_min)
                    all_loo_maxs.append(loo_max)
                else:
                    loo_mean = None
                    loo_std = None
                    loo_min = None
                    loo_max = None
                    num_sentences = len(split_sentences(text))

                # Calculate statistics for cumulative perplexities
                if cumulative_results and len(cumulative_results) > 1:
                    cumulative_perplexities = [r["ppl"] for r in cumulative_results]
                    x = np.arange(1, len(cumulative_perplexities) + 1)
                    y = np.array(cumulative_perplexities)
                    cumulative_slope = np.polyfit(x, y, 1)[0]
                    cumulative_mean = np.mean(cumulative_perplexities)
                    cumulative_std = np.std(cumulative_perplexities)
                    cumulative_first = cumulative_perplexities[0]
                    cumulative_last = cumulative_perplexities[-1]
                    all_cumulative_slopes.append(cumulative_slope)
                else:
                    cumulative_slope = None
                    cumulative_mean = None
                    cumulative_std = None
                    cumulative_first = None
                    cumulative_last = None

                # Calculate statistics for cumulative reversed perplexities
                if cumulative_rev_results and len(cumulative_rev_results) > 1:
                    cumulative_rev_perplexities = [
                        r["ppl"] for r in cumulative_rev_results
                    ]
                    x_rev = np.arange(1, len(cumulative_rev_perplexities) + 1)
                    y_rev = np.array(cumulative_rev_perplexities)
                    cumulative_rev_slope = np.polyfit(x_rev, y_rev, 1)[0]
                    cumulative_rev_mean = np.mean(cumulative_rev_perplexities)
                    cumulative_rev_std = np.std(cumulative_rev_perplexities)
                    cumulative_rev_first = cumulative_rev_perplexities[0]
                    cumulative_rev_last = cumulative_rev_perplexities[-1]
                    all_cumulative_rev_slopes.append(cumulative_rev_slope)
                else:
                    cumulative_rev_slope = None
                    cumulative_rev_mean = None
                    cumulative_rev_std = None
                    cumulative_rev_first = None
                    cumulative_rev_last = None

                # Calculate statistics for cumulative shuffled perplexities
                if cumulative_shuff_results and len(cumulative_shuff_results) > 1:
                    cumulative_shuff_perplexities = [
                        r["ppl"] for r in cumulative_shuff_results
                    ]
                    x_shuff = np.arange(1, len(cumulative_shuff_perplexities) + 1)
                    y_shuff = np.array(cumulative_shuff_perplexities)
                    cumulative_shuff_slope = np.polyfit(x_shuff, y_shuff, 1)[0]
                    cumulative_shuff_mean = np.mean(cumulative_shuff_perplexities)
                    cumulative_shuff_std = np.std(cumulative_shuff_perplexities)
                    cumulative_shuff_first = cumulative_shuff_perplexities[0]
                    cumulative_shuff_last = cumulative_shuff_perplexities[-1]
                    all_cumulative_shuff_slopes.append(cumulative_shuff_slope)
                else:
                    cumulative_shuff_slope = None
                    cumulative_shuff_mean = None
                    cumulative_shuff_std = None
                    cumulative_shuff_first = None
                    cumulative_shuff_last = None

                # Store leave-one-out details for separate CSV
                for loo_item in loo_results:
                    leave_one_out_all.append(
                        {
                            "text_id": key,
                            "sentence_index": loo_item["sentence_index"],
                            "removed_sentence": loo_item["removed_sentence"],
                            "ppl_without_sentence": round(loo_item["ppl"], 2),
                            "original_ppl": round(ppl, 2),
                            "ppl_change": round(loo_item["ppl"] - ppl, 2),
                        }
                    )

                # Store cumulative details for separate CSV
                for cum_item in cumulative_results:
                    cumulative_all.append(
                        {
                            "text_id": key,
                            "num_sentences": cum_item["num_sentences"],
                            "last_sentence": cum_item["last_sentence"],
                            "ppl": round(cum_item["ppl"], 2),
                        }
                    )

                # Store cumulative rev details for separate CSV
                for cum_item in cumulative_rev_results:
                    cumulative_rev_all.append(
                        {
                            "text_id_rev": key,
                            "num_sentences_rev": cum_item["num_sentences"],
                            "last_sentence_rev": cum_item["last_sentence"],
                            "ppl_rev": round(cum_item["ppl"], 2),
                        }
                    )

                # Store cumulative shuffled details for separate CSV
                for cum_shuff_item in cumulative_shuff_results:
                    cumulative_shuff_all.append(
                        {
                            "text_id_shuff": key,
                            "num_sentences_shuff": cum_shuff_item["num_sentences"],
                            "last_sentence_shuff": cum_shuff_item["last_sentence"],
                            "ppl_shuff": round(cum_shuff_item["ppl"], 2),
                        }
                    )

                # Build full results row
                results_list.append(
                    {
                        "id": key,
                        "text": text,
                        "reversed_text": reversed_text,
                        "shuffled_text": shuffled_text,
                        "num_sentences": num_sentences,
                        "ppl": round(ppl, 2),
                        "ppl_reversed": round(ppl_reversed, 2),
                        "ppl_shuffled": round(ppl_shuffled, 2),
                        "ppl_diff_reversed": round(ppl_reversed - ppl, 2),
                        "ppl_diff_shuffled": round(ppl_shuffled - ppl, 2),
                        "ppl_ratio_reversed": (
                            round(ppl_reversed / ppl, 2) if ppl > 0 else None
                        ),
                        "ppl_ratio_shuffled": (
                            round(ppl_shuffled / ppl, 2) if ppl > 0 else None
                        ),
                        "ppl_loo_mean": (
                            round(loo_mean, 2) if loo_mean is not None else None
                        ),
                        "ppl_loo_std": (
                            round(loo_std, 2) if loo_std is not None else None
                        ),
                        "ppl_loo_min": (
                            round(loo_min, 2) if loo_min is not None else None
                        ),
                        "ppl_loo_max": (
                            round(loo_max, 2) if loo_max is not None else None
                        ),
                        "ppl_cumulative_slope": (
                            round(cumulative_slope, 4)
                            if cumulative_slope is not None
                            else None
                        ),
                        "ppl_cumulative_mean": (
                            round(cumulative_mean, 2)
                            if cumulative_mean is not None
                            else None
                        ),
                        "ppl_cumulative_std": (
                            round(cumulative_std, 2)
                            if cumulative_std is not None
                            else None
                        ),
                        "ppl_cumulative_first": (
                            round(cumulative_first, 2)
                            if cumulative_first is not None
                            else None
                        ),
                        "ppl_cumulative_last": (
                            round(cumulative_last, 2)
                            if cumulative_last is not None
                            else None
                        ),
                        "ppl_cumulative_rev_slope": (
                            round(cumulative_rev_slope, 4)
                            if cumulative_rev_slope is not None
                            else None
                        ),
                        "ppl_cumulative_rev_mean": (
                            round(cumulative_rev_mean, 2)
                            if cumulative_rev_mean is not None
                            else None
                        ),
                        "ppl_cumulative_rev_std": (
                            round(cumulative_rev_std, 2)
                            if cumulative_rev_std is not None
                            else None
                        ),
                        "ppl_cumulative_rev_first": (
                            round(cumulative_rev_first, 2)
                            if cumulative_rev_first is not None
                            else None
                        ),
                        "ppl_cumulative_rev_last": (
                            round(cumulative_rev_last, 2)
                            if cumulative_rev_last is not None
                            else None
                        ),
                        "ppl_cumulative_shuff_slope": (
                            round(cumulative_shuff_slope, 4)
                            if cumulative_shuff_slope is not None
                            else None
                        ),
                        "ppl_cumulative_shuff_mean": (
                            round(cumulative_shuff_mean, 2)
                            if cumulative_shuff_mean is not None
                            else None
                        ),
                        "ppl_cumulative_shuff_std": (
                            round(cumulative_shuff_std, 2)
                            if cumulative_shuff_std is not None
                            else None
                        ),
                        "ppl_cumulative_shuff_first": (
                            round(cumulative_shuff_first, 2)
                            if cumulative_shuff_first is not None
                            else None
                        ),
                        "ppl_cumulative_shuff_last": (
                            round(cumulative_shuff_last, 2)
                            if cumulative_shuff_last is not None
                            else None
                        ),
                    }
                )
            else:
                # Build basic results row (only original ppl)
                results_list.append(
                    {
                        "id": key,
                        "text": text,
                        "ppl": round(ppl, 2),
                    }
                )
        else:
            # Handle empty/invalid text
            perplexities[key] = None
            if is_full_analysis:
                perplexities_reversed[key] = None
                perplexities_shuffled[key] = None
                results_list.append(
                    {
                        "id": key,
                        "text": text if isinstance(text, str) else str(text),
                        "reversed_text": None,
                        "shuffled_text": None,
                        "num_sentences": 0,
                        "ppl": None,
                        "ppl_reversed": None,
                        "ppl_shuffled": None,
                        "ppl_diff_reversed": None,
                        "ppl_diff_shuffled": None,
                        "ppl_ratio_reversed": None,
                        "ppl_ratio_shuffled": None,
                        "ppl_loo_mean": None,
                        "ppl_loo_std": None,
                        "ppl_loo_min": None,
                        "ppl_loo_max": None,
                        "ppl_cumulative_slope": None,
                        "ppl_cumulative_mean": None,
                        "ppl_cumulative_std": None,
                        "ppl_cumulative_first": None,
                        "ppl_cumulative_last": None,
                        "ppl_cumulative_rev_slope": None,
                        "ppl_cumulative_rev_mean": None,
                        "ppl_cumulative_rev_std": None,
                        "ppl_cumulative_rev_first": None,
                        "ppl_cumulative_rev_last": None,
                        "ppl_cumulative_shuff_slope": None,
                        "ppl_cumulative_shuff_mean": None,
                        "ppl_cumulative_shuff_std": None,
                        "ppl_cumulative_shuff_first": None,
                        "ppl_cumulative_shuff_last": None,
                    }
                )
            else:
                results_list.append(
                    {
                        "id": key,
                        "text": text if isinstance(text, str) else str(text),
                        "ppl": None,
                    }
                )

    # Calculate statistics
    valid_perplexities = [p for p in ppl_values if p is not None]

    stats = {
        "mean_ori": (
            round(np.mean(valid_perplexities), 2) if valid_perplexities else None
        ),
        "median_ori": (
            round(np.median(valid_perplexities), 2) if valid_perplexities else None
        ),
        "std_ori": round(np.std(valid_perplexities), 2) if valid_perplexities else None,
        "min_ori": round(np.min(valid_perplexities), 2) if valid_perplexities else None,
        "max_ori": round(np.max(valid_perplexities), 2) if valid_perplexities else None,
        "count": len(valid_perplexities),
    }

    if is_full_analysis:
        valid_perplexities_reversed = [p for p in ppl_reversed_values if p is not None]
        valid_perplexities_shuffled = [p for p in ppl_shuffled_values if p is not None]

        stats.update(
            {
                "mean_reversed": (
                    round(np.mean(valid_perplexities_reversed), 2)
                    if valid_perplexities_reversed
                    else None
                ),
                "median_reversed": (
                    round(np.median(valid_perplexities_reversed), 2)
                    if valid_perplexities_reversed
                    else None
                ),
                "std_reversed": (
                    round(np.std(valid_perplexities_reversed), 2)
                    if valid_perplexities_reversed
                    else None
                ),
                "min_reversed": (
                    round(np.min(valid_perplexities_reversed), 2)
                    if valid_perplexities_reversed
                    else None
                ),
                "max_reversed": (
                    round(np.max(valid_perplexities_reversed), 2)
                    if valid_perplexities_reversed
                    else None
                ),
                "mean_shuffled": (
                    round(np.mean(valid_perplexities_shuffled), 2)
                    if valid_perplexities_shuffled
                    else None
                ),
                "median_shuffled": (
                    round(np.median(valid_perplexities_shuffled), 2)
                    if valid_perplexities_shuffled
                    else None
                ),
                "std_shuffled": (
                    round(np.std(valid_perplexities_shuffled), 2)
                    if valid_perplexities_shuffled
                    else None
                ),
                "min_shuffled": (
                    round(np.min(valid_perplexities_shuffled), 2)
                    if valid_perplexities_shuffled
                    else None
                ),
                "max_shuffled": (
                    round(np.max(valid_perplexities_shuffled), 2)
                    if valid_perplexities_shuffled
                    else None
                ),
            }
        )

        # Add aggregate leave-one-out statistics
        if all_loo_means:
            stats["loo_mean_of_means"] = round(np.mean(all_loo_means), 2)
            stats["loo_mean_of_stds"] = round(np.mean(all_loo_stds), 2)
            stats["loo_mean_of_mins"] = round(np.mean(all_loo_mins), 2)
            stats["loo_mean_of_maxs"] = round(np.mean(all_loo_maxs), 2)
            stats["loo_overall_std"] = round(np.std(all_loo_means), 2)
        else:
            stats["loo_mean_of_means"] = None
            stats["loo_mean_of_stds"] = None
            stats["loo_mean_of_mins"] = None
            stats["loo_mean_of_maxs"] = None
            stats["loo_overall_std"] = None

        # Add aggregate cumulative statistics
        if all_cumulative_slopes:
            stats["cumulative_mean_slope"] = round(np.mean(all_cumulative_slopes), 4)
            stats["cumulative_std_slope"] = round(np.std(all_cumulative_slopes), 4)
        else:
            stats["cumulative_mean_slope"] = None
            stats["cumulative_std_slope"] = None

        if all_cumulative_rev_slopes:
            stats["cumulative_rev_mean_slope"] = round(
                np.mean(all_cumulative_rev_slopes), 4
            )
            stats["cumulative_rev_std_slope"] = round(
                np.std(all_cumulative_rev_slopes), 4
            )
        else:
            stats["cumulative_rev_mean_slope"] = None
            stats["cumulative_rev_std_slope"] = None

        if all_cumulative_shuff_slopes:
            stats["cumulative_shuff_mean_slope"] = round(
                np.mean(all_cumulative_shuff_slopes), 4
            )
            stats["cumulative_shuff_std_slope"] = round(
                np.std(all_cumulative_shuff_slopes), 4
            )
        else:
            stats["cumulative_shuff_mean_slope"] = None
            stats["cumulative_shuff_std_slope"] = None

    # Save detailed results to CSV
    df = pd.DataFrame(results_list)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "ppl_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed results to: {csv_path}")

    if is_full_analysis:
        # Save leave-one-out results to separate CSV
        if leave_one_out_all:
            loo_df = pd.DataFrame(leave_one_out_all)
            loo_csv_path = os.path.join(output_dir, "ppl_leave_one_out.csv")
            loo_df.to_csv(loo_csv_path, index=False)
            print(f"Saved leave-one-out results to: {loo_csv_path}")

        # Save cumulative results to separate CSV
        if cumulative_all:
            cumulative_df = pd.DataFrame(cumulative_all)
            cumulative_csv_path = os.path.join(output_dir, "ppl_cumulative.csv")
            cumulative_df.to_csv(cumulative_csv_path, index=False)
            print(f"Saved cumulative results to: {cumulative_csv_path}")

        # Save cumulative rev results to separate CSV
        if cumulative_rev_all:
            cumulative_rev_df = pd.DataFrame(cumulative_rev_all)
            cumulative_rev_csv_path = os.path.join(output_dir, "ppl_cumulative_rev.csv")
            cumulative_rev_df.to_csv(cumulative_rev_csv_path, index=False)
            print(f"Saved cumulative rev results to: {cumulative_rev_csv_path}")

        # Save cumulative shuffled results to separate CSV
        if cumulative_shuff_all:
            cumulative_shuff_df = pd.DataFrame(cumulative_shuff_all)
            cumulative_shuff_csv_path = os.path.join(
                output_dir, "ppl_cumulative_shuffled.csv"
            )
            cumulative_shuff_df.to_csv(cumulative_shuff_csv_path, index=False)
            print(f"Saved cumulative shuffled results to: {cumulative_shuff_csv_path}")

    # Save statistics to CSV
    stats_df = pd.DataFrame([stats])
    stats_csv_path = os.path.join(output_dir, "ppl_statistics.csv")
    stats_df.to_csv(stats_csv_path, index=False)
    print(f"Saved statistics to: {stats_csv_path}")

    return {"perplexities": perplexities, "statistics": stats}


def compute_perplexity_change_ratios(
    json_path, model, tokenizer, output_dir, shuffle_seed=42
):
    """
    Compute perplexity change ratios for original vs shuffled, reversed, and leave-one-out text.

    For each text in the JSON file, calculates:
    - ppl_change_ratio_shuffled = (ppl_shuffled - ppl_original) / ppl_original
    - ppl_change_ratio_reversed = (ppl_reversed - ppl_original) / ppl_original
    - ppl_change_ratio_loo = (ppl_loo_mean - ppl_original) / ppl_original

    Args:
        json_path: Path to JSON file containing texts
        model: Language model for perplexity calculation
        tokenizer: Tokenizer for the model
        output_dir: Directory to save results
        shuffle_seed: Random seed for shuffling (default: 42)

    Returns:
        Dictionary with detailed results and statistics
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    results_list = []
    ppl_original_values = []
    ppl_shuffled_values = []
    ppl_reversed_values = []
    ppl_loo_values = []

    ppl_change_ratios_shuffled = []
    ppl_change_ratios_reversed = []
    ppl_change_ratios_loo = []

    os.makedirs(output_dir, exist_ok=True)

    for key, text in tqdm(
        data.items(), desc=f"Computing change ratios for {Path(json_path).name}"
    ):
        if isinstance(text, str) and text.strip():
            # Calculate original perplexity
            ppl_original = calculate_ppl(text, model, tokenizer)

            # Calculate shuffled perplexity
            shuffled_text = shuffle_sentences(text, seed=shuffle_seed)
            ppl_shuffled = calculate_ppl(shuffled_text, model, tokenizer)

            # Calculate reversed perplexity
            reversed_text = reverse_sentences(text)
            ppl_reversed = calculate_ppl(reversed_text, model, tokenizer)

            # Calculate leave-one-out perplexities and get mean
            loo_results = calculate_ppl_leave_one_out(text, model, tokenizer)
            if loo_results:
                loo_perplexities = [r["ppl"] for r in loo_results]
                ppl_loo_mean = np.mean(loo_perplexities)
            else:
                ppl_loo_mean = None

            # Calculate change ratios
            change_ratio_shuffled = calculate_perplexity_change_ratio(
                ppl_original, ppl_shuffled
            )
            change_ratio_reversed = calculate_perplexity_change_ratio_reversed(
                ppl_original, ppl_reversed
            )
            change_ratio_loo = calculate_perplexity_change_ratio_loo(
                ppl_original, ppl_loo_mean
            )

            # Store individual values if not None
            ppl_original_values.append(ppl_original)
            ppl_shuffled_values.append(ppl_shuffled)
            ppl_reversed_values.append(ppl_reversed)
            if ppl_loo_mean is not None:
                ppl_loo_values.append(ppl_loo_mean)

            # Ratios
            if change_ratio_shuffled is not None:
                ppl_change_ratios_shuffled.append(change_ratio_shuffled)
            if change_ratio_reversed is not None:
                ppl_change_ratios_reversed.append(change_ratio_reversed)
            if change_ratio_loo is not None:
                ppl_change_ratios_loo.append(change_ratio_loo)

            # Store detailed results
            results_list.append(
                {
                    "text_id": key,
                    "num_sentences": len(split_sentences(text)),
                    "ppl_original": round(ppl_original, 4),
                    "ppl_shuffled": (
                        round(ppl_shuffled, 4) if ppl_shuffled is not None else None
                    ),
                    "ppl_reversed": (
                        round(ppl_reversed, 4) if ppl_reversed is not None else None
                    ),
                    "ppl_loo_mean": (
                        round(ppl_loo_mean, 4) if ppl_loo_mean is not None else None
                    ),
                    "ppl_diff_shuffled": (
                        round(ppl_shuffled - ppl_original, 4)
                        if ppl_shuffled is not None
                        else None
                    ),
                    "ppl_diff_reversed": (
                        round(ppl_reversed - ppl_original, 4)
                        if ppl_reversed is not None
                        else None
                    ),
                    "ppl_diff_loo": (
                        round(ppl_loo_mean - ppl_original, 4)
                        if ppl_loo_mean is not None
                        else None
                    ),
                    "ppl_change_ratio_shuffled": (
                        round(change_ratio_shuffled, 6)
                        if change_ratio_shuffled is not None
                        else None
                    ),
                    "ppl_change_ratio_reversed": (
                        round(change_ratio_reversed, 6)
                        if change_ratio_reversed is not None
                        else None
                    ),
                    "ppl_change_ratio_loo": (
                        round(change_ratio_loo, 6)
                        if change_ratio_loo is not None
                        else None
                    ),
                }
            )

    # Calculate statistics
    statistics = {
        "ppl_values": {
            "ori_mean": (
                round(np.mean(ppl_original_values), 4) if ppl_original_values else None
            ),
            "shuffled_mean": (
                round(np.mean(ppl_shuffled_values), 4) if ppl_shuffled_values else None
            ),
            "reversed_mean": (
                round(np.mean(ppl_reversed_values), 4) if ppl_reversed_values else None
            ),
            "loo_mean": (round(np.mean(ppl_loo_values), 4) if ppl_loo_values else None),
        },
        "ratios_shuffled": {
            "mean": (
                round(np.mean(ppl_change_ratios_shuffled), 4)
                if ppl_change_ratios_shuffled
                else None
            ),
            "median": (
                round(np.median(ppl_change_ratios_shuffled), 4)
                if ppl_change_ratios_shuffled
                else None
            ),
            "std": (
                round(np.std(ppl_change_ratios_shuffled), 4)
                if ppl_change_ratios_shuffled
                else None
            ),
            "min": (
                round(np.min(ppl_change_ratios_shuffled), 4)
                if ppl_change_ratios_shuffled
                else None
            ),
            "max": (
                round(np.max(ppl_change_ratios_shuffled), 4)
                if ppl_change_ratios_shuffled
                else None
            ),
            "count": len(ppl_change_ratios_shuffled),
        },
        "ratios_reversed": {
            "mean": (
                round(np.mean(ppl_change_ratios_reversed), 4)
                if ppl_change_ratios_reversed
                else None
            ),
            "median": (
                round(np.median(ppl_change_ratios_reversed), 4)
                if ppl_change_ratios_reversed
                else None
            ),
            "std": (
                round(np.std(ppl_change_ratios_reversed), 4)
                if ppl_change_ratios_reversed
                else None
            ),
            "min": (
                round(np.min(ppl_change_ratios_reversed), 4)
                if ppl_change_ratios_reversed
                else None
            ),
            "max": (
                round(np.max(ppl_change_ratios_reversed), 4)
                if ppl_change_ratios_reversed
                else None
            ),
            "count": len(ppl_change_ratios_reversed),
        },
        "ratios_loo": {
            "mean": (
                round(np.mean(ppl_change_ratios_loo), 4)
                if ppl_change_ratios_loo
                else None
            ),
            "median": (
                round(np.median(ppl_change_ratios_loo), 4)
                if ppl_change_ratios_loo
                else None
            ),
            "std": (
                round(np.std(ppl_change_ratios_loo), 4)
                if ppl_change_ratios_loo
                else None
            ),
            "min": (
                round(np.min(ppl_change_ratios_loo), 4)
                if ppl_change_ratios_loo
                else None
            ),
            "max": (
                round(np.max(ppl_change_ratios_loo), 4)
                if ppl_change_ratios_loo
                else None
            ),
            "count": len(ppl_change_ratios_loo),
        },
    }

    # Save detailed results to CSV
    csv_path = os.path.join(output_dir, "ppl_change_ratios.csv")
    df = pd.DataFrame(results_list)
    df.to_csv(csv_path, index=False)
    print(f"Saved detailed change ratios to: {csv_path}")

    # Save statistics to JSON
    stats_path = os.path.join(output_dir, "ppl_change_ratios_stats.json")
    with open(stats_path, "w") as f:
        json.dump(statistics, f, indent=2)
    print(f"Saved statistics to: {stats_path}")

    return {
        "detailed_results": results_list,
        "statistics": statistics,
    }
