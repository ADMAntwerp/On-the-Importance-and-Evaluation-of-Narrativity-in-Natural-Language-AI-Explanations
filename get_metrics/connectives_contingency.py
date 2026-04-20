import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
import yaml
import json
from utils.connectives_lex import connectives


for resource in ["punkt"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource)

with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    LLM_OUTPUTS_FOLDER = config_data["LLM_OUTPUTS_FOLDER"]
    RESULTS = config_data["RESULTS"]
    TEMPLATE_OUTPUTS_FOLDER = config_data["TEMPLATE_OUTPUTS_FOLDER"]


def compute_connectives_ratio(text, connectives):
    """
    Computes the connectives ratio for a given text (number of connectives / total words).

    Args:
        text (str): The input text.
        connectives (dict): Dictionary of connectives by category.

    Returns:
        float: The connectives ratio.
    """
    tokens = word_tokenize(text.lower())
    # Keep only alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]

    if len(tokens) == 0:
        return 0

    # Create a set of all connectives
    connectives_set = set(connectives.get("contingency", []))

    # Count connectives in the text
    num_connectives = sum(1 for token in tokens if token in connectives_set)
    total_words = len(tokens)

    connectives_ratio = num_connectives / total_words if total_words > 0 else 0

    return connectives_ratio


def main():
    print("Starting connectives contingency ratio computation...\n")
    llm_methods = ["explingo", "explingo_zero_shot", "explingo_narratives", "xaistories", "xaistories_narratives"]
    template_methods = ["talktomodel", "templated_narrative"]
    dataset_names = list(config_data["DATASETS_INFO"].keys())

    for method in template_methods:
            for ds_name in dataset_names:
                results = {}
                print(f"Processing {ds_name} with {method}...")
                input_json = os.path.join(
                    TEMPLATE_OUTPUTS_FOLDER, f"{method}/{ds_name}.json"
                )
                if not os.path.exists(input_json):
                    print(f"File {input_json} does not exist. Skipping.")
                    continue

                # Read the LLM outputs JSON file
                with open(input_json, "r") as f:
                    inputs = json.load(f)

                for i, llm_output in inputs.items():
                    conn_ratio = compute_connectives_ratio(llm_output, connectives)
                    results[i] = {"connectives_ratio": round(conn_ratio, 4)}

                # Save results to a CSV file
                if results:
                    output_csv = os.path.join(
                        RESULTS,
                        f"{method}/{ds_name}/connectives_contingency_ratios_results.csv",
                    )
                    # Create the parent directory, not the full path with filename
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

                    # Convert results to DataFrame
                    df_data = []
                    for instance_id, metrics in results.items():
                        df_data.append(
                            {
                                "instance_index": instance_id,
                                "connectives_ratio": metrics["connectives_ratio"],
                            }
                        )

                    pd.DataFrame(df_data).to_csv(output_csv, index=False)
                    print(f"Connectives ratio results saved to {output_csv}")

    for method in llm_methods:
        for ds_name in dataset_names:
                results = {}
                print(f"Processing {ds_name} with {method}...")
                input_json = os.path.join(
                    LLM_OUTPUTS_FOLDER, f"{method}/{ds_name}.json"
                )
                if not os.path.exists(input_json):
                    print(f"File {input_json} does not exist. Skipping.")
                    continue

                # Read the LLM outputs JSON file
                with open(input_json, "r") as f:
                    inputs = json.load(f)

                for i, llm_output in inputs.items():
                    conn_ratio = compute_connectives_ratio(llm_output, connectives)
                    results[i] = {"connectives_ratio": round(conn_ratio, 4)}

                # Save results to a CSV file
                if results:
                    output_csv = os.path.join(
                        RESULTS,
                        f"{method}/{ds_name}/connectives_contingency_ratios_results.csv",
                    )
                    # Create the parent directory, not the full path with filename
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

                    # Convert results to DataFrame
                    df_data = []
                    for instance_id, metrics in results.items():
                        df_data.append(
                            {
                                "instance_index": instance_id,
                                "connectives_ratio": metrics["connectives_ratio"],
                            }
                        )

                    pd.DataFrame(df_data).to_csv(output_csv, index=False)
                    print(f"Connectives ratio results saved to {output_csv}")


if __name__ == "__main__":
    main()
