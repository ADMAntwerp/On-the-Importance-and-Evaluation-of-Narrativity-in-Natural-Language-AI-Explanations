import os
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import yaml
import json

# Ensure necessary NLTK data is downloaded
for resource in ["punkt", "stopwords", "opinion_lexicon"]:
    try:
        if resource == "opinion_lexicon":
            nltk.data.find("corpora/opinion_lexicon")
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


def compute_ttr(text, remove_stopwords=False):
    """
    Computes the Type-Token Ratio (TTR) for a given text.

    Args:
        text (str): The text to analyze.
        remove_stopwords (bool): Whether to remove stop words before computing TTR.

    Returns:
        float: The TTR value.
    """
    tokens = word_tokenize(text.lower())
    # Keep only alphabetic tokens
    tokens = [token for token in tokens if token.isalpha()]

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [token for token in tokens if token not in stop_words]

    total_tokens = len(tokens)
    unique_types = len(set(tokens))
    ttr = unique_types / total_tokens if total_tokens > 0 else 0
    return ttr


def main():
    print("Starting TTR computation...\n")
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
                    ttr = compute_ttr(llm_output, remove_stopwords=False)
                    results[i] = {"TTR": round(ttr, 2)}

                # Save results to a CSV file
                if results:
                    output_csv = os.path.join(
                        RESULTS, f"{method}/{ds_name}/lexical_diversity_ttr_results.csv"
                    )
                    # Create the parent directory, not the full path with filename
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

                    # Convert results to DataFrame
                    df_data = []
                    for instance_id, metrics in results.items():
                        df_data.append(
                            {"instance_index": instance_id, "TTR": metrics["TTR"]}
                        )

                    pd.DataFrame(df_data).to_csv(output_csv, index=False)
                    print(f"TTR results saved to {output_csv}")

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
                    ttr = compute_ttr(llm_output, remove_stopwords=False)
                    results[i] = {"TTR": round(ttr, 2)}

                # Save results to a CSV file
                if results:
                    output_csv = os.path.join(
                        RESULTS, f"{method}/{ds_name}/lexical_diversity_ttr_results.csv"
                    )
                    # Create the parent directory, not the full path with filename
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

                    # Convert results to DataFrame
                    df_data = []
                    for instance_id, metrics in results.items():
                        df_data.append(
                            {"instance_index": instance_id, "TTR": metrics["TTR"]}
                        )

                    pd.DataFrame(df_data).to_csv(output_csv, index=False)
                    print(f"TTR results saved to {output_csv}")

    print("TTR computation completed.\n")


if __name__ == "__main__":
    main()
