"""
Script to generate the perplexity summary tables and charts from new results format.

New result format structure (per method/dataset):
- cumulative_ppl_analysis.json: Contains dcpr_metric, fitted_parameters, fit_quality, etc.
- ppl_change_ratios_stats.json: Contains ppl_values and ratio statistics
- ppl_change_ratios.csv: Detailed per-text perplexity change ratios
- cem_detailed.csv: Contains cem and perplexity per instance
"""

import json
import os
import pandas as pd
import yaml
from prettytable import PrettyTable


with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    DATASETS_FOLDER = config_data["DATASET_FOLDER"]
    RESULTS = config_data["RESULTS"]
    SUMMARY_FOLDER = config_data["SUMMARY_FOLDER"]


def df_to_pretty(df: pd.DataFrame, title: str | None = None) -> PrettyTable:
    """Convert a pandas DataFrame to a PrettyTable."""
    t = PrettyTable()
    if title:
        t.title = title
    t.field_names = list(df.columns)
    t.add_rows(df.to_records(index=False).tolist())
    t.align = "l"
    t.max_width = 28
    return t


def get_dataset_names(folder_path):
    """Extract dataset names from CSV files in the folder."""
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist, using default datasets")
        return ["compas", "diabetes", "fifa", "german_credit", "stroke", "student"]

    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    dataset_names = [os.path.splitext(f)[0] for f in csv_files]
    return sorted(dataset_names)


def load_cumulative_ppl_analysis(results_folder):
    """
    Load cumulative perplexity analysis from the new JSON format.
    
    Returns dict with structure: {method/dataset: {dcpr_metric, fitted_parameters, ...}}
    """
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
            json_path = os.path.join(method_path, dataset, "cumulative_ppl_analysis.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    data[f"{method}/{dataset}"] = json.load(f)
    
    return data


def load_ppl_change_ratios_stats(results_folder):
    """
    Load perplexity change ratio statistics from the new JSON format.
    
    Returns dict with structure: {method/dataset: {ppl_values, ratios_shuffled, ...}}
    """
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


def create_dcpr_summary_table(cumulative_data):
    """
    Create a summary table for dcpr metrics from cumulative PPL analysis.
    """
    rows = []
    for key, values in cumulative_data.items():
        method, dataset = key.split("/")
        if "error" in values:
            continue
        
        row = {
            "Dataset": dataset,
            "Method": method,
            "DCPR": round(values.get("dcpr_metric", 0), 2),
            "R²": round(values.get("fit_quality", {}).get("r_squared", 0), 2),
            "RMSE": round(values.get("fit_quality", {}).get("rmse", 0), 2),
            "Num_texts": values.get("dataset_statistics", {}).get("num_texts", "NA"),
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Dataset", "Method"]).reset_index(drop=True)
    return df


def create_ppl_change_ratios_table(change_ratios_data):
    """
    Create a summary table for perplexity change ratios.
    Ratios are calculated as: (PPL_variant - PPL_ori) / PPL_ori
    """
    rows = []
    for key, values in change_ratios_data.items():
        method, dataset = key.split("/")
        
        ppl_values = values.get("ppl_values", {})
        
        # Round PPL values first
        ppl_ori = round(ppl_values.get("ori_mean", 0), 2)
        ppl_shuf = round(ppl_values.get("shuffled_mean", 0), 2)
        ppl_rev = round(ppl_values.get("reversed_mean", 0), 2)
        ppl_loo = round(ppl_values.get("loo_mean", 0), 2)
        
        # Calculate ratios from rounded PPL values: (variant - original) / original
        ratio_shuf = round((ppl_shuf - ppl_ori) / ppl_ori, 2) if ppl_ori > 0 else 0
        ratio_rev = round((ppl_rev - ppl_ori) / ppl_ori, 2) if ppl_ori > 0 else 0
        ratio_loo = round((ppl_loo - ppl_ori) / ppl_ori, 2) if ppl_ori > 0 else 0
        
        row = {
            "Dataset": dataset,
            "Method": method,
            "PPL_ori": ppl_ori,
            "PPL_shuf": ppl_shuf,
            "PPL_rev": ppl_rev,
            "PPL_loo": ppl_loo,
            "Ratio_shuf": ratio_shuf,
            "Ratio_rev": ratio_rev,
            "Ratio_loo": ratio_loo,
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["Dataset", "Method"]).reset_index(drop=True)
    return df


def generate_dcpr_latex_table(dcpr_df: pd.DataFrame, output_file: str, one_sentence_methods: set = None) -> None:
    """
    Generate a LaTeX table for DCPR metrics.

    Excludes one-sentence methods and bolds best values per metric per dataset.
    Metrics: DCPR (down), R² (up), RMSE (down)
    """
    if one_sentence_methods is None:
        one_sentence_methods = {"talktomodel", "explingo"}

    # Filter out one-sentence methods
    display_df = dcpr_df[~dcpr_df["Method"].isin(one_sentence_methods)].copy()
    display_df = display_df.sort_values(["Dataset", "Method"]).reset_index(drop=True)

    metric_specs = [
        ("DCPR", r"DCPR $\downarrow$", "down"),
        ("R²", r"$R^2$ $\uparrow$", "up"),
        ("RMSE", r"RMSE $\downarrow$", "down"),
    ]

    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\caption{Fitted $\rho$ example of DCPR metrics across datasets and methods.}")
    lines.append(r"\label{tab:fitted_dcpr}")
    lines.append(r"\begin{tabular}{llrrr}")
    lines.append(r"\toprule")

    # Header
    header = r"Dataset & Method & DCPR $\downarrow$ & $R^2$ $\uparrow$ & RMSE $\downarrow$ \\"
    lines.append(header)
    lines.append(r"\midrule")

    n_datasets = display_df["Dataset"].nunique()
    for dataset_idx, (dataset_name, dataset_block) in enumerate(display_df.groupby("Dataset", sort=False)):
        dataset_block = dataset_block.reset_index(drop=True)
        n_methods = len(dataset_block)

        # Calculate best values per metric in this dataset
        best_values = {}
        for metric_key, _, direction in metric_specs:
            if metric_key in display_df.columns:
                values = dataset_block[metric_key].dropna()
                if len(values) > 0:
                    if direction == "down":
                        best_values[metric_key] = values.min()
                    else:  # "up"
                        best_values[metric_key] = values.max()

        for row_idx in range(n_methods):
            row = dataset_block.iloc[row_idx]
            if row_idx == 0:
                dataset_cell = f"\\multirow{{{n_methods}}}{{*}}{{{dataset_name.capitalize()}}}"
            else:
                dataset_cell = ""

            cells = [dataset_cell, row["Method"]]

            for metric_key, _, _ in metric_specs:
                if metric_key not in display_df.columns:
                    continue

                value = row[metric_key]
                if pd.isna(value):
                    cells.append(r"\multicolumn{1}{c}{--}")
                else:
                    formatted = f"{value:.2f}"
                    # Bold if this is the best value for this metric in this dataset
                    if metric_key in best_values and float(value) == best_values[metric_key]:
                        formatted = rf"\textbf{{{formatted}}}"
                    cells.append(formatted)

            lines.append(" & ".join(cells) + r" \\")

        if dataset_idx < n_datasets - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    os.makedirs(f"{SUMMARY_FOLDER}/tables", exist_ok=True)
    
    # Load cumulative PPL analysis data
    print("Loading cumulative PPL analysis data...")
    cumulative_data = load_cumulative_ppl_analysis(RESULTS)
    
    if cumulative_data:
        dcpr_df = create_dcpr_summary_table(cumulative_data)
        pretty_dcpr = df_to_pretty(dcpr_df, title="Fitted \rho example of DCPR Metrics")
        print(pretty_dcpr)

        output_file = f"{SUMMARY_FOLDER}/tables/dcpr_summary_table.txt"
        with open(output_file, "w") as f:
            f.write(str(pretty_dcpr))
        print(f"\nDCPR summary table saved to '{output_file}'\n")

        # Generate LaTeX table for DCPR
        latex_file = f"{SUMMARY_FOLDER}/tables/dcpr_summary_table.tex"
        generate_dcpr_latex_table(dcpr_df, latex_file)
        print(f"DCPR LaTeX table saved to '{latex_file}'\n")
    else:
        print("No cumulative PPL analysis data found.\n")
    
    # Load PPL change ratios data
    print("Loading PPL change ratios data...")
    change_ratios_data = load_ppl_change_ratios_stats(RESULTS)
    
    if change_ratios_data:
        ratios_df = create_ppl_change_ratios_table(change_ratios_data)
        pretty_ratios = df_to_pretty(ratios_df, title="PPL Change Ratios Summary")
        print(pretty_ratios)
        
        output_file = f"{SUMMARY_FOLDER}/tables/ppl_change_ratios_table.txt"
        with open(output_file, "w") as f:
            f.write(str(pretty_ratios))
        print(f"\nPPL change ratios table saved to '{output_file}'\n")
