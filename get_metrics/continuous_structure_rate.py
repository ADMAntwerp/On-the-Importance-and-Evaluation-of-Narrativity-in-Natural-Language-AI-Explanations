import yaml
import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from utils.perplexity_utils import compute_perplexity_change_ratios


with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    LLM_OUTPUTS_FOLDER = config_data["LLM_OUTPUTS_FOLDER"]
    TEMPLATE_OUTPUTS_FOLDER = config_data["TEMPLATE_OUTPUTS_FOLDER"]
    MODEL_PPL = config_data["MODEL_PPL"]
    RESULTS = config_data["RESULTS"]
    SUMMARY_FOLDER = config_data["SUMMARY_FOLDER"]
    ONE_SENTENCE_EXPLAINATION_METHODS = config_data["ONE_SENTENCE_EXPLAINATION_METHODS"]

# Manual exclusion list - add method names here to skip them from analysis
MANUAL_EXCLUSION_METHODS = []

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
    print("Starting perplexity change ratio computation...\n")
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

            # Methods with single-sentence explanations - completely skip them
            # This analysis is only for multi-sentence narratives
            skip_methods = [
                method.lower() for method in ONE_SENTENCE_EXPLAINATION_METHODS
            ]
            # Add manual exclusions
            skip_methods.extend([method.lower() for method in MANUAL_EXCLUSION_METHODS])

            if method_name.lower() in skip_methods:
                print(f"  Skipping {method_name} (excluded from analysis)")
                continue

            # Compute perplexity change ratios
            results = compute_perplexity_change_ratios(
                json_file, model, tokenizer, output_dir
            )
            folder_results[identifier] = results

            # Print statistics
            stats = results["statistics"]
            print(f"\nChange Ratio Statistics for {identifier}:")
            print(f"  PPL Values:")
            print(f"    Original Text Mean PPL: {stats['ppl_values']['ori_mean']:.2f}")
            print(
                f"    Shuffled Text Mean PPL: {stats['ppl_values']['shuffled_mean']:.2f}"
            )
            print(
                f"    Reversed Text Mean PPL: {stats['ppl_values']['reversed_mean']:.2f}"
            )
            print(
                f"    Leave-One-Out Text Mean PPL: {stats['ppl_values']['loo_mean']:.2f}"
            )
            print(f"  Shuffled Text Change Ratio:")
            print(f"    Mean: {stats['ratios_shuffled']['mean']:.4f}")
            print(f"    Median: {stats['ratios_shuffled']['median']:.4f}")
            print(f"    Std Dev: {stats['ratios_shuffled']['std']:.4f}")
            print(f"    Min: {stats['ratios_shuffled']['min']:.4f}")
            print(f"    Max: {stats['ratios_shuffled']['max']:.4f}")
            print(f"    Count: {stats['ratios_shuffled']['count']}")
            print(f"  Reversed Text Change Ratio:")
            print(f"    Mean: {stats['ratios_reversed']['mean']:.4f}")
            print(f"    Median: {stats['ratios_reversed']['median']:.4f}")
            print(f"    Std Dev: {stats['ratios_reversed']['std']:.4f}")
            print(f"    Min: {stats['ratios_reversed']['min']:.4f}")
            print(f"    Max: {stats['ratios_reversed']['max']:.4f}")
            print(f"    Count: {stats['ratios_reversed']['count']}")
            print(f"  Leave-One-Out Change Ratio:")
            print(f"    Mean: {stats['ratios_loo']['mean']:.4f}")
            print(f"    Median: {stats['ratios_loo']['median']:.4f}")
            print(f"    Std Dev: {stats['ratios_loo']['std']:.4f}")
            print(f"    Min: {stats['ratios_loo']['min']:.4f}")
            print(f"    Max: {stats['ratios_loo']['max']:.4f}")
            print(f"    Count: {stats['ratios_loo']['count']}")

        all_results[folder_type] = folder_results

    # Save overall summary
    output_path = os.path.join(SUMMARY_FOLDER, "perplexity_change_ratios")
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "change_ratios_summary.json")

    # Extract statistics for summary
    summary = {}
    for folder_type, folder_results in all_results.items():
        summary[folder_type] = {}
        for identifier, results in folder_results.items():
            summary[folder_type][identifier] = results["statistics"]

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Overall summary saved to: {output_file}")
    print(f"{'='*60}\n")
    print("Perplexity change ratio computation completed.\n")


if __name__ == "__main__":
    main()
