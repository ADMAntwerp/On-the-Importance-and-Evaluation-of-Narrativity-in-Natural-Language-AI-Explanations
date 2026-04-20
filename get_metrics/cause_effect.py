import yaml
import json
import os
import re
import torch
import pandas as pd
from runpy import run_path
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.perplexity_utils import compute_cumulative_ppl_and_dcpr


with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    LLM_OUTPUTS_FOLDER = config_data["LLM_OUTPUTS_FOLDER"]
    TEMPLATE_OUTPUTS_FOLDER = config_data["TEMPLATE_OUTPUTS_FOLDER"]
    MODEL_PPL = config_data["MODEL_PPL"]
    RESULTS = config_data["RESULTS"]
    SUMMARY_FOLDER = config_data["SUMMARY_FOLDER"]

# Initialize model and tokenizer
print(f"Loading model: {MODEL_PPL}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PPL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PPL,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
model.eval()

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def load_cause_effect_markers():
    """Load cause-effect markers from utils/cause-effect_lex.py."""
    lexicon_path = Path("utils") / "cause-effect_lex.py"
    lexicon_globals = run_path(str(lexicon_path))
    markers = lexicon_globals.get("cause_effect_markers", [])
    return [m.strip().lower() for m in markers if isinstance(m, str) and m.strip()]


def compute_cause_effect_ratio(text, markers):
    """Compute cause-effect ratio = marker occurrences / total words."""
    if not isinstance(text, str) or not text.strip():
        return 0.0

    text_lower = text.lower()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text_lower)
    total_words = len(tokens)

    if total_words == 0:
        return 0.0

    marker_count = 0
    for marker in markers:
        pattern = r"\b" + re.escape(marker).replace(r"\ ", r"\s+") + r"\b"
        marker_count += len(re.findall(pattern, text_lower))

    return marker_count / total_words


def main():
    print("Starting cause-effect ratio and CECPR computation...\n")
    os.makedirs(RESULTS, exist_ok=True)
    cause_effect_markers = load_cause_effect_markers()

    folders_to_process = {
        "llm_outputs": LLM_OUTPUTS_FOLDER,
        "template_outputs": TEMPLATE_OUTPUTS_FOLDER,
    }

    all_results = {}

    for folder_type, folder_path in folders_to_process.items():
        print(f"\n{'='*60}")
        print(f"Processing {folder_type}: {folder_path}")
        print(f"{'='*60}\n")

        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping.")
            continue

        json_files = list(Path(folder_path).rglob("*.json"))

        if not json_files:
            print(f"No JSON files found in {folder_path}")
            continue

        folder_results = {}

        for json_file in json_files:
            relative_path = json_file.relative_to(folder_path)
            method_name = (
                relative_path.parent.name if relative_path.parent.name else "root"
            )
            dataset_name = json_file.stem
            identifier = f"{method_name}/{dataset_name}"

            print(f"\nProcessing: {identifier}")

            output_dir = os.path.join(RESULTS, method_name, dataset_name)

            with open(json_file, "r") as f:
                dataset_outputs = json.load(f)

            ce_ratios = []
            ce_details = []
            for instance_id, generated_text in dataset_outputs.items():
                ce_ratio = compute_cause_effect_ratio(generated_text, cause_effect_markers)
                ce_ratios.append(ce_ratio)
                ce_details.append(
                    {
                        "instance_index": instance_id,
                        "cause_effect_ratio": round(ce_ratio, 6),
                    }
                )

            avg_ce_ratio = float(sum(ce_ratios) / len(ce_ratios)) if ce_ratios else 0.0

            ce_ratio_csv = os.path.join(output_dir, "cause_effect_ratios_results.csv")
            os.makedirs(output_dir, exist_ok=True)
            pd.DataFrame(ce_details).to_csv(ce_ratio_csv, index=False)

            # Compute cumulative perplexity analysis and fitted r parameter.
            results = compute_cumulative_ppl_and_dcpr(
                json_file, model, tokenizer, output_dir, method_name, dataset_name
            )

            if "error" not in results:
                results.setdefault("dataset_statistics", {})[
                    "avg_cause_effect_ratio"
                ] = round(avg_ce_ratio, 6)

                r_value = results.get("fitted_parameters", {}).get("r")
                if r_value is not None and avg_ce_ratio > 0:
                    cecpr = r_value / (avg_ce_ratio ** 2)
                    results["cecpr_metric"] = round(cecpr, 3)
                else:
                    results["cecpr_metric"] = None

                with open(os.path.join(output_dir, "cumulative_ppl_analysis.json"), "w") as f:
                    json.dump(results, f, indent=2)

            folder_results[identifier] = results

            # Print results
            if "error" in results:
                print(f"  Error: {results['error']}")
            else:
                params = results["fitted_parameters"]
                fit_quality = results.get("fit_quality", {})
                stats = results["dataset_statistics"]
                cecpr = results.get("cecpr_metric")

                print(f"\n  Fitted Exponential Model: y(x) = A * r^x")
                if params["fit_success"]:
                    print(f"    A (baseline perplexity): {params['A']}")
                    print(f"    r (decay rate): {params['r']}")
                else:
                    print(f"    Fit failed")

                if fit_quality:
                    print(f"\n  Fit Quality Metrics:")
                    print(
                        f"    R² (coefficient of determination): {fit_quality['r_squared']:.6f}"
                    )
                    print(
                        f"    RMSE (root mean squared error): {fit_quality['rmse']:.6f}"
                    )
                    print(
                        f"    MAPE (mean absolute percentage error): {fit_quality['mape']:.2f}%"
                    )
                    print(f"    Residual mean: {fit_quality['residual_mean']:.6f}")
                    print(f"    Residual std: {fit_quality['residual_std']:.6f}")

                print(f"\n  Dataset Statistics:")
                print(f"    Number of texts: {stats['num_texts']}")
                print(f"    Sentence count range: {stats['num_sentences_range']}")
                print(
                    f"    Avg cause-effect ratio: {stats['avg_cause_effect_ratio']:.6f}"
                )

                if cecpr is not None:
                    print(f"\n  CECPR Metric (r / avg_cause_effect_ratio^2): {cecpr:.3f}")
                else:
                    print("\n  CECPR Metric: None (missing fitted r or zero ratio)")

        all_results[folder_type] = folder_results

    # Save overall summary
    output_path = os.path.join(SUMMARY_FOLDER, "continuity_cumulative_ppl_cecpr")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "cecpr_summary.json")

    # Extract main results for summary
    summary = {}
    for folder_type, folder_results in all_results.items():
        summary[folder_type] = {}
        for identifier, results in folder_results.items():
            if "error" not in results:
                summary[folder_type][identifier] = {
                    "fitted_parameters": results["fitted_parameters"],
                    "fit_quality": results.get("fit_quality", {}),
                    "cecpr_metric": results.get("cecpr_metric"),
                    "dataset_statistics": results["dataset_statistics"],
                }
            else:
                summary[folder_type][identifier] = results

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Overall summary saved to: {output_file}")
    print(f"{'='*60}\n")
    print("Cause-effect ratio and CECPR computation completed.\n")


if __name__ == "__main__":
    main()
