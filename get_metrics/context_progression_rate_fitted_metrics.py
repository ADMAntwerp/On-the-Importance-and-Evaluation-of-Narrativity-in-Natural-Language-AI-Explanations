import yaml
import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
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


def main():
    print("Starting cumulative perplexity and DCPR metric computation...\n")
    os.makedirs(RESULTS, exist_ok=True)

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

            # Compute cumulative perplexity analysis and dcpr metric
            results = compute_cumulative_ppl_and_dcpr(
                json_file, model, tokenizer, output_dir, method_name, dataset_name
            )
            folder_results[identifier] = results

            # Print results
            if "error" in results:
                print(f"  Error: {results['error']}")
            else:
                params = results["fitted_parameters"]
                fit_quality = results.get("fit_quality", {})
                stats = results["dataset_statistics"]
                dcpr = results["dcpr_metric"]

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
                    f"    Avg distinct 2-gram ratio: {stats['avg_distinct_2gram_ratio']:.6f}"
                )

                print(f"\n  DCPR Metric (r / avg_distinct_2gram_ratio): {dcpr:.3f}")

        all_results[folder_type] = folder_results

    # Save overall summary
    output_path = os.path.join(SUMMARY_FOLDER, "continuity_cumulative_ppl_dcpr")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "dcpr_summary.json")

    # Extract main results for summary
    summary = {}
    for folder_type, folder_results in all_results.items():
        summary[folder_type] = {}
        for identifier, results in folder_results.items():
            if "error" not in results:
                summary[folder_type][identifier] = {
                    "fitted_parameters": results["fitted_parameters"],
                    "fit_quality": results.get("fit_quality", {}),
                    "dcpr_metric": results["dcpr_metric"],
                    "dataset_statistics": results["dataset_statistics"],
                }
            else:
                summary[folder_type][identifier] = results

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Overall summary saved to: {output_file}")
    print(f"{'='*60}\n")
    print("Cumulative perplexity and DCPR metric computation completed.\n")


if __name__ == "__main__":
    main()
