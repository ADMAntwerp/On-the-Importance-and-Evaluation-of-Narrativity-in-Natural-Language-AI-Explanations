import os
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import yaml
import json
import re

# Ensure necessary NLTK data is downloaded
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


def count_syllables(word):
    """
    Count syllables in a word using a simple heuristic.
    """
    word = word.lower()
    # Remove non-alphabetic characters
    word = re.sub(r"[^a-z]", "", word)
    if not word:
        return 0

    # Count vowel groups
    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel

    # Handle silent 'e' at the end
    if word.endswith("e") and syllable_count > 1:
        syllable_count -= 1

    # Every word has at least one syllable
    return max(1, syllable_count)


def count_complex_words(words):
    """
    Count words with 3 or more syllables (complex words).
    """
    complex_count = 0
    for word in words:
        if count_syllables(word) >= 3:
            complex_count += 1
    return complex_count


def compute_gunning_fog_index(text):
    """
    Computes the Gunning Fog Index for readability.
    Formula: 0.4 * ((words/sentences) + 100 * (complex_words/words))
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0

    words = word_tokenize(text.lower())
    # Keep only alphabetic tokens
    words = [word for word in words if word.isalpha()]

    if not words:
        return 0.0

    num_sentences = len(sentences)
    num_words = len(words)
    num_complex_words = count_complex_words(words)

    avg_sentence_length = num_words / num_sentences
    complex_word_ratio = (num_complex_words / num_words) * 100

    gunning_fog = 0.4 * (avg_sentence_length + complex_word_ratio)
    return gunning_fog


def compute_flesch_reading_ease(text):
    """
    Computes the Flesch Reading Ease score.
    Formula: 206.835 - (1.015 * ASL) - (84.6 * ASW)
    Where ASL = Average Sentence Length, ASW = Average Syllables per Word
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0

    words = word_tokenize(text.lower())
    # Keep only alphabetic tokens
    words = [word for word in words if word.isalpha()]

    if not words:
        return 0.0

    num_sentences = len(sentences)
    num_words = len(words)
    total_syllables = sum(count_syllables(word) for word in words)

    avg_sentence_length = num_words / num_sentences
    avg_syllables_per_word = total_syllables / num_words

    flesch_score = (
        206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    )
    return flesch_score


def compute_flesch_kincaid_grade_level(text):
    """
    Computes the Flesch-Kincaid Grade Level.
    Formula: (0.39 * ASL) + (11.8 * ASW) - 15.59
    Where ASL = Average Sentence Length, ASW = Average Syllables per Word
    """
    sentences = sent_tokenize(text)
    if not sentences:
        return 0.0

    words = word_tokenize(text.lower())
    # Keep only alphabetic tokens
    words = [word for word in words if word.isalpha()]

    if not words:
        return 0.0

    num_sentences = len(sentences)
    num_words = len(words)
    total_syllables = sum(count_syllables(word) for word in words)

    avg_sentence_length = num_words / num_sentences
    avg_syllables_per_word = total_syllables / num_words

    fk_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
    return fk_grade


def compute_all_readability_metrics(text):
    """
    Compute all readability metrics for a given text.
    """
    gunning_fog = compute_gunning_fog_index(text)
    flesch_ease = compute_flesch_reading_ease(text)
    flesch_kincaid = compute_flesch_kincaid_grade_level(text)

    return (round(gunning_fog, 2), round(flesch_ease, 2), round(flesch_kincaid, 2))


def main():
    methods = ["xaistories", "explingo", "talktomodel"]
    dataset_names = list(config_data["DATASETS_INFO"].keys())

    for method in methods:
        if method == "talktomodel":
            for ds_name in dataset_names:
                results = {}
                print(f"Processing {ds_name} with {method}...")
                input_json = os.path.join(
                    TEMPLATE_OUTPUTS_FOLDER, f"{method}/{ds_name}_templates.json"
                )
                if not os.path.exists(input_json):
                    print(f"File {input_json} does not exist. Skipping.")
                    continue

                # Read the LLM outputs JSON file
                with open(input_json, "r") as f:
                    inputs = json.load(f)

                for i, llm_output in inputs.items():
                    gunning_fog, flesch_ease, flesch_kincaid = (
                        compute_all_readability_metrics(llm_output)
                    )
                    results[i] = {
                        "gunning_fog_index": gunning_fog,
                        "flesch_reading_ease": flesch_ease,
                        "flesch_kincaid_grade_level": flesch_kincaid,
                    }

                # Save results to a CSV file
                if results:
                    output_csv = os.path.join(
                        RESULTS, f"{method}/{ds_name}/readability_metrics.csv"
                    )
                    # Create the parent directory, not the full path with filename
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

                    # Convert results to DataFrame
                    df_data = []
                    for instance_id, metrics in results.items():
                        df_data.append(
                            {
                                "instance_index": instance_id,
                                "gunning_fog_index": metrics["gunning_fog_index"],
                                "flesch_reading_ease": metrics["flesch_reading_ease"],
                                "flesch_kincaid_grade_level": metrics[
                                    "flesch_kincaid_grade_level"
                                ],
                            }
                        )

                    pd.DataFrame(df_data).to_csv(output_csv, index=False)
                    print(f"Readability metrics results saved to {output_csv}")

        if method in ["xaistories", "explingo"]:
            for ds_name in dataset_names:
                results = {}
                print(f"Processing {ds_name} with {method}...")
                input_json = os.path.join(
                    LLM_OUTPUTS_FOLDER, f"{method}/{ds_name}_llm_outputs.json"
                )
                if not os.path.exists(input_json):
                    print(f"File {input_json} does not exist. Skipping.")
                    continue

                # Read the LLM outputs JSON file
                with open(input_json, "r") as f:
                    inputs = json.load(f)

                for i, llm_output in inputs.items():
                    gunning_fog, flesch_ease, flesch_kincaid = (
                        compute_all_readability_metrics(llm_output)
                    )
                    results[i] = {
                        "gunning_fog_index": gunning_fog,
                        "flesch_reading_ease": flesch_ease,
                        "flesch_kincaid_grade_level": flesch_kincaid,
                    }

                # Save results to a CSV file
                if results:
                    output_csv = os.path.join(
                        RESULTS, f"{method}/{ds_name}/readability_metrics.csv"
                    )
                    # Create the parent directory, not the full path with filename
                    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

                    # Convert results to DataFrame
                    df_data = []
                    for instance_id, metrics in results.items():
                        df_data.append(
                            {
                                "instance_index": instance_id,
                                "gunning_fog_index": metrics["gunning_fog_index"],
                                "flesch_reading_ease": metrics["flesch_reading_ease"],
                                "flesch_kincaid_grade_level": metrics[
                                    "flesch_kincaid_grade_level"
                                ],
                            }
                        )

                    pd.DataFrame(df_data).to_csv(output_csv, index=False)
                    print(f"Readability metrics results saved to {output_csv}")


if __name__ == "__main__":
    main()
