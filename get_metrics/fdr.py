"""Fluency-Diversity Rate (FDR) metrics: FDR score, distinct 2-gram ratio, perplexity."""

import yaml
import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.perplexity_utils import compute_fdr_for_json


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
    print("Starting FDR computation...\n")
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

            results = compute_fdr_for_json(
                json_file, model, tokenizer, output_dir
            )
            folder_results[identifier] = results

            print(f"\nStatistics for {identifier}:")
            print(
                f"  FDR - Mean: {results['statistics']['fdr']['mean']:.4f}, "
                f"Std: {results['statistics']['fdr']['std']:.4f}"
            )
            print(
                f"  Distinct 2-gram ratio - Mean: {results['statistics']['distinct_bigram_ratio']['mean']:.4f}, "
                f"Std: {results['statistics']['distinct_bigram_ratio']['std']:.4f}"
            )
            print(
                f"  Perplexity - Mean: {results['statistics']['perplexity']['mean']:.4f}, "
                f"Std: {results['statistics']['perplexity']['std']:.4f}"
            )

        all_results[folder_type] = folder_results

    # Save overall summary
    output_path = os.path.join(SUMMARY_FOLDER, "fdr")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "fdr_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Overall results saved to: {output_file}")
    print(f"{'='*60}\n")

    # Create summary statistics file
    summary = {}
    for folder_type, folder_results in all_results.items():
        summary[folder_type] = {}
        for identifier, results in folder_results.items():
            summary[folder_type][identifier] = results["statistics"]

    summary_file = os.path.join(output_path, "fdr_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Overall summary saved to: {summary_file}\n")
    print("FDR computation completed.\n")


if __name__ == "__main__":
    main()
