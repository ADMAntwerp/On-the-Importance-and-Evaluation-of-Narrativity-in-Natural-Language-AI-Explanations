"""
Script to generate the summary table for NLP metrics from new results format.
"""

import json
import os
import pandas as pd
import yaml
from prettytable import PrettyTable
from typing import Dict


with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    RESULTS = config_data["RESULTS"]
    SUMMARY_FOLDER = config_data["SUMMARY_FOLDER"]
    ONE_SENTENCE_METHODS = set(config_data.get("ONE_SENTENCE_EXPLAINATION_METHODS", []))


CONTINUITY_METRICS = ["CSR", "CECPR", "DCPR", "CCPR", "TTCPR", "VCPR"]


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


def _escape_latex(text: str) -> str:
    """Escape LaTeX special characters in plain text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    escaped = text
    for old, new in replacements.items():
        escaped = escaped.replace(old, new)
    return escaped


def _format_number(value: float, decimals: int = 2) -> str:
    """Format numeric values with fixed precision for table output."""
    return f"{value:.{decimals}f}"


def _rank_to_color(rank: float, n_methods: int) -> str:
    """Map rank to a red-green color scale (best=green, worst=red)."""
    if n_methods <= 1:
        green, red = 80, 20
    else:
        ratio = (rank - 1) / (n_methods - 1)
        green = int(round((1 - ratio) * 80))
        red = int(round(ratio * 80))
    return f"green!{green}!red!{red}"


def _compute_metric_ranks(
    summary_df: pd.DataFrame, metric_directions: Dict[str, str], method_col: str = "Method"
) -> pd.DataFrame:
    """Compute per dataset per metric ranks across methods."""
    ranks = summary_df[["Dataset", method_col]].copy()

    for metric, direction in metric_directions.items():
        if metric not in summary_df.columns:
            continue
        ascending = direction == "down"
        ranks[metric] = (
            summary_df.groupby("Dataset")[metric]
            .rank(ascending=ascending, method="min")
            .astype(float)
        )

    return ranks


def compute_average_ranks(summary_df: pd.DataFrame, metric_directions: Dict[str, str]) -> pd.DataFrame:
    """Compute average method rank across all dataset-metric pairs (Demsar style)."""
    rank_df = _compute_metric_ranks(summary_df, metric_directions)
    metric_cols = [m for m in metric_directions if m in rank_df.columns]

    long_df = rank_df.melt(
        id_vars=["Dataset", "Method"], value_vars=metric_cols, var_name="Metric", value_name="Rank"
    )
    long_df = long_df.dropna(subset=["Rank"])

    avg_rank = (
        long_df.groupby("Method", as_index=False)["Rank"]
        .mean()
        .rename(columns={"Rank": "Average_Rank"})
        .sort_values("Average_Rank", ascending=True)
        .reset_index(drop=True)
    )
    avg_rank["Average_Rank"] = avg_rank["Average_Rank"].round(3)
    return avg_rank


def get_metric_groups() -> Dict[str, Dict[str, str]]:
    """Metric groups for separate ranking/CD analysis."""
    return {
        "set1_main": {
            "PPL": "down",
            "Bigram_ratio": "up",
            "TTR": "up",
            "Verb_ratio": "up",
            "Conn_all": "up",
        },
        "set2_discourse": {
            "FDR": "up",
            "CSR": "up",
            "CECPR": "down",
            "DCPR": "down",
            "CCPR": "down",
            "TTCPR": "down",
            "VCPR": "down",
        },
    }


def get_group_method_exclusions() -> Dict[str, set[str]]:
    """Method exclusions applied per metric group for ranking/CD analysis."""
    return {
        "set1_main": set(),
        "set2_discourse": {"talktomodel", "explingo"},
    }


def generate_colored_latex_table(summary_df: pd.DataFrame, output_file: str) -> None:
    """
    Generate a LaTeX table with dataset-wise rankings and bold best values.

    Required LaTeX packages:
    - booktabs
    - multirow
    """
    metric_specs = [
        ("PPL", r"PPL $\downarrow$", "down"),
        ("Bigram_ratio", r"$\text{Dist}_2$ $\uparrow$", "up"),
        ("TTR", r"TTR $\uparrow$", "up"),
        ("Verb_ratio", r"VR $\uparrow$", "up"),
        ("Conn_all", r"CD $\uparrow$", "up"),
        ("FDR", r"\emph{FDR} $\uparrow$", "up"),
        ("CSR", r"\emph{CSR} $\uparrow$", "up"),
        ("CECPR", r"\emph{CECPR} $\downarrow$", "down"),
        ("DCPR", r"\emph{DCPR} $\downarrow$", "down"),
        ("CCPR", r"\emph{CCPR} $\downarrow$", "down"),
        ("TTCPR", r"\emph{TTCPR} $\downarrow$", "down"),
        ("VCPR", r"\emph{VCPR} $\downarrow$", "down"),
    ]
    metric_directions = {k: d for k, _, d in metric_specs if k in summary_df.columns}

    rank_df = _compute_metric_ranks(summary_df, metric_directions)

    display_df = summary_df.copy()
    display_df = display_df.sort_values(["Dataset", "Method"]).reset_index(drop=True)
    rank_df = rank_df.sort_values(["Dataset", "Method"]).reset_index(drop=True)

    lines = []
    lines.append(r"\begin{table}[th!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Summary of evaluation metrics for the generated explanations across six tabular datasets. "
        r"Arrows in the column headers indicate the optimal direction for each metric. "
        r"Bold values denote the best score for each metric across a dataset. "
        r"Average ranks are exported separately for Demsar-style critical difference analysis.}"
    )
    lines.append(r"\label{tab:summary_results}")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.8}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(r"    \begin{tabular}{llccccc|ccccccc}")
    lines.append(r"    \toprule")

    header = (
        r"Dataset & Method & "
        + " & ".join(h for k, h, _ in metric_specs if k in display_df.columns)
        + r"\\"
    )
    lines.append(f"    {header}")
    lines.append(r"    \midrule")

    for dataset_idx, (dataset_name, dataset_block) in enumerate(display_df.groupby("Dataset", sort=False)):
        dataset_block = dataset_block.reset_index(drop=True)
        rank_block = rank_df[rank_df["Dataset"] == dataset_name].reset_index(drop=True)
        n_methods = len(dataset_block)

        for row_idx in range(n_methods):
            row = dataset_block.iloc[row_idx]
            row_rank = rank_block.iloc[row_idx]

            if row_idx == 0:
                dataset_cell = f"\multirow{{{n_methods}}}{{*}}{{{_escape_latex(str(dataset_name).capitalize())}}}"
            else:
                dataset_cell = ""

            method_cell = _escape_latex(str(row["Method"]))
            cells = [dataset_cell, method_cell]

            for metric_key, _, _ in metric_specs:
                if metric_key not in display_df.columns:
                    continue

                value = row[metric_key]
                rank = row_rank.get(metric_key)

                if pd.isna(value) or pd.isna(rank):
                    cells.append("-")
                    continue

                formatted = _format_number(float(value), decimals=2)
                if float(rank) == 1.0:
                    formatted = rf"\textbf{{{formatted}}}"

                cells.append(formatted)

            lines.append("    " + " & ".join(cells) + r"\\")

        if dataset_idx < display_df["Dataset"].nunique() - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def generate_big_main_latex_table(summary_df: pd.DataFrame, output_file: str) -> None:
    """
    Generate a BIG main LaTeX table from the complete summary dataframe.

    This export is uncolored and intended as the primary full-results table.
    Required LaTeX packages:
    - booktabs
    - multirow
    """
    metric_specs = [
        ("PPL", r"PPL $\downarrow$", "down"),
        ("Bigram_ratio", r"$\text{Dist}_2$ $\uparrow$", "up"),
        ("TTR", r"TTR $\uparrow$", "up"),
        ("Verb_ratio", r"VR $\uparrow$", "up"),
        ("Conn_all", r"CD $\uparrow$", "up"),
        ("FDR", r"\emph{FDR} $\uparrow$", "up"),
        ("CSR", r"\emph{CSR} $\uparrow$", "up"),
        ("CECPR", r"\emph{CECPR} $\downarrow$", "down"),
        ("DCPR", r"\emph{DCPR} $\downarrow$", "down"),
        ("CCPR", r"\emph{CCPR} $\downarrow$", "down"),
        ("TTCPR", r"\emph{TTCPR} $\downarrow$", "down"),
        ("VCPR", r"\emph{VCPR} $\downarrow$", "down"),
    ]
    metric_specs = [(k, h, d) for k, h, d in metric_specs if k in summary_df.columns]

    display_df = summary_df.sort_values(["Dataset", "Method"]).reset_index(drop=True)

    # ll + one centered numeric column per metric
    col_spec = "ll" + ("c" * len(metric_specs))

    lines = []
    lines.append(r"\begin{table}[th!]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{BIG main summary table of evaluation metrics for generated explanations across datasets and methods.}"
    )
    lines.append(r"\label{tab:summary_results_big_main}")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{0.85}")
    lines.append(r"\resizebox{\textwidth}{!}{")
    lines.append(f"    \begin{{tabular}}{{{col_spec}}}")
    lines.append(r"    \toprule")

    header = r"Dataset & Method"
    if metric_specs:
        header += " & " + " & ".join(h for _, h, _ in metric_specs)
    header += r"\\"
    lines.append(f"    {header}")
    lines.append(r"    \midrule")

    n_datasets = display_df["Dataset"].nunique()
    for dataset_idx, (dataset_name, dataset_block) in enumerate(display_df.groupby("Dataset", sort=False)):
        dataset_block = dataset_block.reset_index(drop=True)
        n_methods = len(dataset_block)

        # For each metric, find the best value in this dataset
        best_values = {}
        for metric_key, _, direction in metric_specs:
            values = dataset_block[metric_key].dropna()
            if len(values) > 0:
                if direction == "down":
                    best_values[metric_key] = values.min()
                else:  # "up"
                    best_values[metric_key] = values.max()

        for row_idx in range(n_methods):
            row = dataset_block.iloc[row_idx]
            if row_idx == 0:
                dataset_cell = f"\multirow{{{n_methods}}}{{*}}{{{_escape_latex(str(dataset_name).capitalize())}}}"
            else:
                dataset_cell = ""

            method_cell = _escape_latex(str(row["Method"]))
            cells = [dataset_cell, method_cell]

            for metric_key, _, _ in metric_specs:
                value = row[metric_key]
                if pd.isna(value):
                    cells.append("-")
                else:
                    formatted = _format_number(float(value), decimals=2)
                    # Bold if this is the best value for this metric in this dataset
                    if metric_key in best_values and float(value) == best_values[metric_key]:
                        formatted = rf"\textbf{{{formatted}}}"
                    cells.append(formatted)

            lines.append("    " + " & ".join(cells) + r"\\")

        if dataset_idx < n_datasets - 1:
            lines.append(r"    \midrule")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"}")
    lines.append(r"\end{table}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def create_summary_table(results_folder):
    """
    Creates a summary table aggregating results from the new metric file formats.
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

            # connectives_all_ratios_results.csv (new format)
            conn_all_path = os.path.join(dataset_path, "connectives_all_ratios_results.csv")
            if os.path.exists(conn_all_path):
                df_conn = pd.read_csv(conn_all_path)
                row["Conn_all"] = round(df_conn["connectives_ratio"].mean(), 2)
                row["Conn_all_std"] = round(df_conn["connectives_ratio"].std(), 2)
            
            # connectives_contingency_ratios_results.csv (new format)
            conn_cont_path = os.path.join(dataset_path, "connectives_contingency_ratios_results.csv")
            if os.path.exists(conn_cont_path):
                df_conn_cont = pd.read_csv(conn_cont_path)
                row["Conn_cont"] = round(df_conn_cont["connectives_ratio"].mean(), 2)
                row["Conn_cont_std"] = round(df_conn_cont["connectives_ratio"].std(), 2)
            
            # connectives_expansion_ratios_results.csv (new format)
            conn_exp_path = os.path.join(dataset_path, "connectives_expansion_ratios_results.csv")
            if os.path.exists(conn_exp_path):
                df_conn_exp = pd.read_csv(conn_exp_path)
                row["Conn_exp"] = round(df_conn_exp["connectives_ratio"].mean(), 2)
                row["Conn_exp_std"] = round(df_conn_exp["connectives_ratio"].std(), 2)

            # lexical_diversity_ttr_results.csv (new format)
            ttr_path = os.path.join(dataset_path, "lexical_diversity_ttr_results.csv")
            if os.path.exists(ttr_path):
                df_ttr = pd.read_csv(ttr_path)
                row["TTR"] = round(df_ttr["TTR"].mean(), 2)
                row["TTR_std"] = round(df_ttr["TTR"].std(), 2)

            # lexical_diversity_verb_ratios_results.csv (new format)
            verb_path = os.path.join(dataset_path, "lexical_diversity_verb_ratios_results.csv")
            if os.path.exists(verb_path):
                df_verb = pd.read_csv(verb_path)
                row["Verb_ratio"] = round(df_verb["verb_ratio"].mean(), 2)
                row["Verb_ratio_std"] = round(df_verb["verb_ratio"].std(), 2)
            
            # fdc_detailed.csv (new format, with backward compatibility)
            fdc_path = os.path.join(dataset_path, "fdr_detailed.csv")
            cem_path = os.path.join(dataset_path, "cem_detailed.csv")
            metric_path = fdc_path if os.path.exists(fdc_path) else cem_path
            if os.path.exists(metric_path):
                df_metric = pd.read_csv(metric_path)
                metric_col = "fdc" if "fdc" in df_metric.columns else "cem"
                row["FDR"] = round(df_metric[metric_col].mean(), 2)
                row["FDR_std"] = round(df_metric[metric_col].std(), 2)
                row["PPL"] = round(df_metric["perplexity"].mean(), 2)
                row["PPL_std"] = round(df_metric["perplexity"].std(), 2)
                row["Bigram_ratio"] = round(df_metric["distinct_bigram_ratio"].mean(), 2)

            # ppl_change_ratios_stats.json (Ratio_shuf)
            ppl_ratios_path = os.path.join(dataset_path, "ppl_change_ratios_stats.json")
            if os.path.exists(ppl_ratios_path):
                with open(ppl_ratios_path, "r") as f:
                    ppl_ratios = json.load(f)
                ppl_values = ppl_ratios.get("ppl_values", {})
                ppl_ori = ppl_values.get("ori_mean", 0)
                ppl_shuf = ppl_values.get("shuffled_mean", 0)
                if ppl_ori > 0:
                    row["CSR"] = round((ppl_shuf - ppl_ori) / ppl_ori, 2)

            # cumulative_ppl_analysis.json (WAR metrics)
            analysis_path = os.path.join(dataset_path, "cumulative_ppl_analysis.json")
            if os.path.exists(analysis_path):
                with open(analysis_path, "r") as f:
                    analysis = json.load(f)
                avg_ce_ratio = analysis.get("dataset_statistics", {}).get("avg_cause_effect_ratio")
                # if avg_ce_ratio is not None:
                #    row["CE_ratio"] = round(avg_ce_ratio, 2)
                if analysis.get("cecpr_metric") is not None:
                    row["CECPR"] = round(analysis["cecpr_metric"], 2)
                if analysis.get("dcpr_metric") is not None:
                    row["DCPR"] = round(analysis["dcpr_metric"], 2)
                if analysis.get("ccpr_metric") is not None:
                    row["CCPR"] = round(analysis["ccpr_metric"], 2)
                if analysis.get("ttcpr_metric") is not None:
                    row["TTCPR"] = round(analysis["ttcpr_metric"], 2)
                if analysis.get("vcpr_metric") is not None:
                    row["VCPR"] = round(analysis["vcpr_metric"], 2)

            # One-sentence methods are not comparable on continuity metrics.
            if method in ONE_SENTENCE_METHODS:
                for metric_name in CONTINUITY_METRICS:
                    row[metric_name] = pd.NA

            summary_data.append(row)

    # DataFrame
    summary_df = pd.DataFrame(summary_data)

    cols = [
        "Dataset",
        "Method",
        "PPL",
        # "PPL_std",
        "Bigram_ratio",
        "Conn_all",       
        "TTR",
        # "TTR_std",
        "Verb_ratio",
        # "Verb_ratio_std",
        # "Conn_all_std",
        "FDR",
        # "FDR_std",
        "CSR",
        # "CE_ratio",
        "CECPR",
        "DCPR",
        "CCPR",
        "TTCPR",
        "VCPR",
    ]
    present_cols = [c for c in cols if c in summary_df.columns]
    summary_df = summary_df.reindex(columns=present_cols)
    
    # Sort by Dataset first, then Method in custom order
    method_order = [
        "talktomodel",
        "templated_narrative",
        "explingo",
        "explingo_zero_shot",
        "explingo_narratives",
        "xaistories",
        "xaistories_narratives",
    ]
    # Exclude xainarratives
    summary_df = summary_df[summary_df["Method"] != "xainarratives"]
    summary_df["Method"] = pd.Categorical(summary_df["Method"], categories=method_order, ordered=True)
    summary_df = summary_df.sort_values(["Dataset", "Method"]).reset_index(drop=True)

    return summary_df




if __name__ == "__main__":
    summary = create_summary_table(results_folder=RESULTS)

    pretty_table = df_to_pretty(summary, title="Summary Results")
    print(pretty_table)

    output_path = os.path.join(SUMMARY_FOLDER, "tables")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "summary_table.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(pretty_table) + "\n")

    print(f"\nSummary table saved to '{output_file}'\n")

    # Generate LaTeX table (without colors)
    latex_file = os.path.join(output_path, "summary_table.tex")
    generate_big_main_latex_table(summary, latex_file)
    print(f"LaTeX summary saved to '{latex_file}'")

    # Export average ranks separately for the two metric sets
    metric_groups = get_metric_groups()
    method_exclusions = get_group_method_exclusions()
    for group_name, group_metrics in metric_groups.items():
        rank_metrics = {k: v for k, v in group_metrics.items() if k in summary.columns}
        if not rank_metrics:
            print(f"Skipping average rank export for '{group_name}' (no matching metrics).")
            continue

        excluded = method_exclusions.get(group_name, set())
        summary_for_group = summary[~summary["Method"].isin(excluded)].copy()
        if summary_for_group.empty:
            print(f"Skipping average rank export for '{group_name}' (no methods left after exclusion).")
            continue

        avg_rank_df = compute_average_ranks(summary_for_group, rank_metrics)

        avg_rank_csv = os.path.join(output_path, f"summary_average_ranks_{group_name}.csv")
        avg_rank_df.to_csv(avg_rank_csv, index=False)
        print(f"Average ranks ({group_name}) saved to '{avg_rank_csv}'")

        avg_rank_latex = os.path.join(output_path, f"summary_average_ranks_{group_name}.tex")
        avg_rank_df.to_latex(avg_rank_latex, index=False, float_format="%.3f")
        print(f"Average ranks LaTeX ({group_name}) saved to '{avg_rank_latex}'")
