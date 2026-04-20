import os
import json
import pandas as pd
import yaml
from utils.prompt_utils import prompt_configs, list_binary


with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)


def get_feature_description(config_feature_desc, feature_name, description=True):
    """Return the description for a given feature from the prompt config."""
    for feat in config_feature_desc:
        if description:
            if feature_name in feat:
                return feat[feature_name]
        else:
            return feature_name
    return ""


def generate_explanations_json(
    ds_name: str,
    base_folder: str,
    output_path: str | None = None,
    max_features: int = 5,
    use_shap_index_as_idx: bool = True,
    sentence_per_feature: bool = False,
):
    """
    Reads (under base_folder):
      - {ds_name}_test_sample.csv   (must contain 'predicted_y')
      - {ds_name}_shap_values.csv   (SHAP values by feature; first column is index)

    Writes a JSON file of:
      [ {"idx": <row id>, "explanation": <markdown>}, ... ]

    Explanation template per row:
    - For the instance {idx}, the model {prediction}:
        {1st feature} is the most important feature and has a {positive/negative} influence on the prediction,
        {2nd feature} is the second most important feature and has a {positive/negative} influence on the prediction,
        {3rd feature} is the third most important feature and has a {positive/negative} influence on the prediction,
        (up to 5 features total)
    """

    def _ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            suf = "th"
        else:
            suf = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
        return f"{n}{suf}"

    test_path = os.path.join(base_folder, f"{ds_name}_test_sample.csv")
    shap_path = os.path.join(base_folder, f"{ds_name}_shap_values.csv")

    test_df = pd.read_csv(test_path)  # keep original row order
    shap_df_full = pd.read_csv(shap_path, index_col=0)  # keep index labels

    # common feature set (ignore 'predicted_y' and any non-overlapping cols)
    common_feats = [
        c for c in shap_df_full.columns if c in test_df.columns and c != "predicted_y"
    ]

    shap_df = (
        shap_df_full[common_feats].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    )

    prompt_config = prompt_configs[ds_name]
    config_feature_desc = prompt_config["feature_desc"]

    # align lengths
    n = min(len(test_df), len(shap_df))
    k = min(max_features, 5)

    records = {}
    for i in range(n):
        idx_value = shap_df.index[i] if use_shap_index_as_idx else i
        # Convert to regular Python int to avoid JSON serialization issues
        prediction = test_df.at[i, "predicted_y"]

        # (feature, shap) pairs for this row
        pairs = [
            (feat, float(shap_df.iat[i, j])) for j, feat in enumerate(common_feats)
        ]
        pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        top = pairs[:k]

        bullets = []
        for rank, (feat_name, shap_val) in enumerate(top, start=1):
            sign = "positive" if shap_val >= 0 else "negative"

            # Get feature description
            desc = get_feature_description(config_feature_desc, feat_name)

            # Determine the rank text
            # Like here https://github.com/dylan-slack/TalkToModel/blob/73612ebb3e72f4f8172380bab8c7ba941e70224b/explain/explanation.py#L430
            # https://github.com/dylan-slack/TalkToModel/blob/73612ebb3e72f4f8172380bab8c7ba941e70224b/explain/explanation.py#L473
            # https://github.com/dylan-slack/TalkToModel/blob/73612ebb3e72f4f8172380bab8c7ba941e70224b/explain/actions/predict.py#L23
            if rank == 1:
                rank_text = "most important"
            elif rank == 2:
                rank_text = "second most important"
            elif rank == 3:
                rank_text = "third most important"
            else:
                rank_text = f"{_ordinal(rank)} most important"

            if not sentence_per_feature:  # TalkToModel: all features in one sentence
                bullets.append(
                    f"{str(feat_name).capitalize()} ({desc}) is the {rank_text} feature "
                    f"and has a {sign} influence on the prediction"
                )
                explanation = (
                    f"For the instance {idx_value}, the model {str(prediction).lower()}: "
                    + (
                        ", ".join(bullets)
                        if bullets
                        else "* (No important features found)"
                    )
                    + "."
                )
            else:  # Templated_Narrative: one feature per sentence
                bullets.append(
                    f" {str(feat_name).capitalize()} ({desc}) is the {rank_text} feature "
                    f"and has a {sign} influence on the prediction"
                )
                explanation = (
                    f"For this instance, the model {str(prediction).lower()}."
                    + (
                        ".".join(bullets)
                        if bullets
                        else " (No important features found)"
                    )
                    + "."
                )

        records[i] = explanation

    return records


def main():

    with open("config.yaml", "r") as f:
        config_data = yaml.safe_load(f)
    TOP_FEATURES = int(config_data["TOP_FEATURES"])
    OUTPUT_FOLDER = config_data["TEMPLATE_OUTPUTS_FOLDER"]
    BASE_FOLDER = config_data["SHAP_RESULTS_FOLDER"]

    methods = ["talktomodel", "templated_narrative"]

    for method in methods:
        if method == "talktomodel":
            # ensure folders exist
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            method_out_dir = os.path.join(OUTPUT_FOLDER, method)
            os.makedirs(method_out_dir, exist_ok=True)

            for ds_name, _ in config_data["DATASETS_INFO"].items():
                templates = generate_explanations_json(
                    ds_name=ds_name,
                    base_folder=BASE_FOLDER,
                    output_path=os.path.join(method_out_dir, f"{ds_name}.json"),
                    max_features=TOP_FEATURES,
                    sentence_per_feature=False,
                )
                # all_inputs[ds_name] = templates
                output_dir = os.path.join(OUTPUT_FOLDER, method)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{ds_name}.json")
                with open(output_file, "w") as f:
                    json.dump(templates, f, indent=4, default=str)
                print(f"TalkToModel templates for {ds_name} saved to {output_file}")

        elif method == "templated_narrative":
            # ensure folders exist
            os.makedirs(OUTPUT_FOLDER, exist_ok=True)
            method_out_dir = os.path.join(OUTPUT_FOLDER, method)
            os.makedirs(method_out_dir, exist_ok=True)

            for ds_name, _ in config_data["DATASETS_INFO"].items():
                templates = generate_explanations_json(
                    ds_name=ds_name,
                    base_folder=BASE_FOLDER,
                    output_path=os.path.join(method_out_dir, f"{ds_name}.json"),
                    max_features=4,
                    sentence_per_feature=True,
                )
                # all_inputs[ds_name] = templates
                output_dir = os.path.join(OUTPUT_FOLDER, method)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{ds_name}.json")
                with open(output_file, "w") as f:
                    json.dump(templates, f, indent=4, default=str)
                print(
                    f"Templated_Narrative templates for {ds_name} saved to {output_file}"
                )


if __name__ == "__main__":
    main()
