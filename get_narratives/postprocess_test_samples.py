import os
import pandas as pd

DATA_TO_PROCESS = "data/SHAP_results"

# Define value mappings for each dataset
VALUE_MAPPINGS = {
    "german_credit": {
        "Job": {
            0: "unskilled and non-resident",
            1: "unskilled and resident",
            2: "skilled",
            3: "highly skilled",
        }
    },
    "student": {
        "school": {"GP": "Gabriel Pereira", "MS": "Mousinho da Silveir"},
        "address": {"U": "urban", "R": "rural"},
        "famsize": {"LE3": "less or equal to 3", "GT3": "greater than 3"},
        "Medu": {
            0: "none",
            1: "primary education (4th grade)",
            2: "5th to 9th grade",
            3: "secondary education",
            4: "higher education",
        },
        "Fedu": {
            0: "none",
            1: "primary education (4th grade)",
            2: "5th to 9th grade",
            3: "secondary education",
            4: "higher education",
        },
        "traveltime": {1: "<15 min.", 2: "15-30 min.", 3: "30-60 min.", 4: ">1 hour"},
        "studytime": {1: "<2 hours", 2: "2-5 hours", 3: "5-10 hours", 4: ">10 hours"},
        "famrel": {1: "very bad", 2: "bad", 3: "average", 4: "good", 5: "excellent"},
        "freetime": {1: "very low", 2: "low", 3: "average", 4: "high", 5: "very high"},
        "goout": {1: "very low", 2: "low", 3: "average", 4: "high", 5: "very high"},
        "Dalc": {1: "very low", 2: "low", 3: "average", 4: "high", 5: "very high"},
        "Walc": {1: "very low", 2: "low", 3: "average", 4: "high", 5: "very high"},
        "health": {1: "very bad", 2: "bad", 3: "average", 4: "good", 5: "very good"},
        "Pstatus": {"T": "living together", "A": "living apart"},
    },
    "stroke": {
        "hypertension": {0: "No", 1: "Yes"},
        "heart_disease": {0: "No", 1: "Yes"},
    },
}

# List of datasets to postprocess that need mapping changes
DATASETS_TO_MAP = ["german_credit", "student", "stroke"]


def apply_mappings(df, mappings):
    """
    For each column in the mappings dictionary, if the column exists in df,
    replace its values according to the mapping.
    """
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)
    return df


def main():
    for dataset in DATASETS_TO_MAP:
        file_path = os.path.join(DATA_TO_PROCESS, f"{dataset}_test_sample.csv")
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping {dataset}.")
            continue

        print(f"Processing {dataset}...")
        df = pd.read_csv(file_path)

        # Apply the mapping for the dataset
        if dataset in VALUE_MAPPINGS:
            df = apply_mappings(df, VALUE_MAPPINGS[dataset])

        # Save the updated file (overwrite or new file)
        df.to_csv(file_path, index=False)
        print(f"Updated test sample saved to {file_path}\n")


if __name__ == "__main__":
    main()
