"""
Script to generate comparison plots across all datasets/methods for cumulative perplexity analysis.
Reads from the cumulative_ppl_analysis.json outputs and creates comparative visualizations.
"""

import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.perplexity_utils import plot_cumulative_ppl_comparison


def main():
    """
    Aggregate cumulative PPL analysis results and generate comparison plots.
    """
    # Paths
    repo_root = Path(__file__).parent.parent
    results_dir = repo_root / "results"
    summary_dir = repo_root / "summary" / "cumulative_ppl_comparison"

    os.makedirs(summary_dir, exist_ok=True)

    # Collect all results
    all_results = {}

    # Iterate through results directory to find all cumulative_ppl_analysis.json files
    for method_dir in results_dir.iterdir():
        if not method_dir.is_dir():
            continue

        method_name = method_dir.name

        for dataset_dir in method_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            analysis_file = dataset_dir / "cumulative_ppl_analysis.json"

            if analysis_file.exists():
                try:
                    with open(analysis_file, "r") as f:
                        results = json.load(f)

                    identifier = f"{method_name}/{dataset_name}"
                    all_results[identifier] = results
                    print(f"✓ Loaded: {identifier}")

                except Exception as e:
                    print(f"✗ Error loading {identifier}: {e}")
            else:
                print(f"⊘ Not found: {identifier}")

    if not all_results:
        print("\nNo cumulative_ppl_analysis.json files found!")
        print(f"Searched in: {results_dir}")
        sys.exit(1)

    print(f"\nTotal results found: {len(all_results)}")
    print("\nGenerating comparison plots...")

    # Generate comparison plots
    plot_cumulative_ppl_comparison(all_results, str(summary_dir))

    print(f"\n✓ Comparison plots saved to: {summary_dir}")

    # Print summary statistics
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    dcpr_values = {}
    r_values = {}
    a_values = {}

    for identifier, results in all_results.items():
        if "error" not in results and results["fitted_parameters"]["fit_success"]:
            dcpr_values[identifier] = results["dcpr_metric"]
            r_values[identifier] = results["fitted_parameters"]["r"]
            a_values[identifier] = results["fitted_parameters"]["A"]

    if dcpr_values:
        print("\ndcpr Metric (Continuity Score) - Lower is Better:")
        sorted_dcpr = sorted(dcpr_values.items(), key=lambda x: x[1])
        for identifier, dcpr in sorted_dcpr:
            print(f"  {identifier:40s}: {dcpr:.6f}")

        print(f"\nBest (lowest dcpr):  {sorted_dcpr[0][0]} ({sorted_dcpr[0][1]:.6f})")
        print(f"Worst (highest dcpr): {sorted_dcpr[-1][0]} ({sorted_dcpr[-1][1]:.6f})")

    if r_values:
        print("\nGrowth Rate (r) - Lower is Better (less growth):")
        sorted_r = sorted(r_values.items(), key=lambda x: x[1])
        for identifier, r in sorted_r:
            print(f"  {identifier:40s}: {r:.6f}")

        print(f"\nBest (lowest r):  {sorted_r[0][0]} ({sorted_r[0][1]:.6f})")
        print(f"Worst (highest r): {sorted_r[-1][0]} ({sorted_r[-1][1]:.6f})")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
