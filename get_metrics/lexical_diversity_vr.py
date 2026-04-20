import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import yaml
import json

# Ensure necessary NLTK data is downloaded
for resource in ["punkt", "averaged_perceptron_tagger", "stopwords"]:
    try:
        if resource == "averaged_perceptron_tagger":
            nltk.data.find("taggers/averaged_perceptron_tagger")
        elif resource == "punkt":
            nltk.data.find("tokenizers/punkt")
        elif resource == "stopwords":
            nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download(resource)

with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    LLM_OUTPUTS_FOLDER = config_data["LLM_OUTPUTS_FOLDER"]
    RESULTS = config_data["RESULTS"]
    TEMPLATE_OUTPUTS_FOLDER = config_data["TEMPLATE_OUTPUTS_FOLDER"]


def compute_verb_ratio(text):
    """
    Computes the verb ratio for a given text (number of verbs / total words).

    Args:
        text (str): The text to analyze.

    Returns:
        float: The verb ratio value.
    """
    tokens = word_tokenize(text.lower())
    # Keep only alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]

    if len(tokens) == 0:
        return 0

    # POS tag the tokens
    pos_tags = pos_tag(tokens)

    # Count verbs (VB, VBD, VBG, VBN, VBP, VBZ)
    verb_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    num_verbs = sum(1 for word, tag in pos_tags if tag in verb_tags)

    total_words = len(tokens)
    verb_ratio = num_verbs / total_words if total_words > 0 else 0

    return verb_ratio


def main():
    print("Starting verb ratio computation...\n")
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
                    verb_ratio = compute_verb_ratio(llm_output)
                    results[i] = {"verb_ratio": round(verb_ratio, 4)}

                # Save results to a CSV file
                if results:
                    output_csv = os.path.join(
                        RESULTS,
                        f"{method}/{ds_name}/lexical_diversity_verb_ratios_results.csv",
                    )
                    # Create the parent directory, not the full path with filename
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

                    # Convert results to DataFrame
                    df_data = []
                    for instance_id, metrics in results.items():
                        df_data.append(
                            {
                                "instance_index": instance_id,
                                "verb_ratio": metrics["verb_ratio"],
                            }
                        )

                    pd.DataFrame(df_data).to_csv(output_csv, index=False)
                    print(f"Verb ratio results saved to {output_csv}")

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
                    verb_ratio = compute_verb_ratio(llm_output)
                    results[i] = {"verb_ratio": round(verb_ratio, 4)}

                # Save results to a CSV file
                if results:
                    output_csv = os.path.join(
                        RESULTS,
                        f"{method}/{ds_name}/lexical_diversity_verb_ratios_results.csv",
                    )
                    # Create the parent directory, not the full path with filename
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

                    # Convert results to DataFrame
                    df_data = []
                    for instance_id, metrics in results.items():
                        df_data.append(
                            {
                                "instance_index": instance_id,
                                "verb_ratio": metrics["verb_ratio"],
                            }
                        )

                    pd.DataFrame(df_data).to_csv(output_csv, index=False)
                    print(f"Verb ratio results saved to {output_csv}")

    print("Verb ratio computation completed.\n")


if __name__ == "__main__":
    main()
