import os
import json
import pandas as pd
import numpy as np
import yaml
import os
import pandas as pd
from math import inf
from utils.prompt_utils import prompt_configs, list_binary


with open("config.yaml", "r") as f:
    config_data = yaml.safe_load(f)
    TOP_FEATURES = int(config_data["TOP_FEATURES"])


def get_feature_description(config_feature_desc, feature_name, description=True):
    """Return the description for a given feature from the prompt config."""
    for feat in config_feature_desc:
        if description:
            if feature_name in feat:
                return feat[feature_name]
        else:
            return feature_name
    return ""


def xaistories_prepare_inputs(dataset_name, top_features=TOP_FEATURES):
    """
    Prepare LLM input instances from SHAP results and test samples for a given dataset.
    Returns a list of input dictionaries and the method name used.
    """
    inputs = []

    base_folder = config_data["SHAP_RESULTS_FOLDER"]
    test_sample_path = os.path.join(base_folder, f"{dataset_name}_test_sample.csv")
    shap_values_path = os.path.join(base_folder, f"{dataset_name}_shap_values.csv")

    test_df = pd.read_csv(test_sample_path)
    shap_df = pd.read_csv(shap_values_path, index_col=0)

    prompt_config = prompt_configs[dataset_name]

    task_description = prompt_config["task_description"]
    dataset_description = prompt_config["dataset_description"]
    target_description = prompt_config["target_description"]
    config_feature_desc = prompt_config["feature_desc"]

    # Compute feature averages from the test sample (numeric only)
    test_features = test_df.copy()
    if "predicted_y" in test_features.columns:
        test_features = test_features.drop(columns=["predicted_y"])
    feature_avgs = round(test_features.mean(numeric_only=True), 2)

    # Reset indices to ensure alignment between test_df and shap_df
    test_df = test_df.reset_index(drop=True)
    shap_df = shap_df.reset_index(drop=True)

    for idx in test_df.index:
        predicted_class = test_df.loc[idx, "predicted_y"]
        feature_rows = []
        for feat in shap_df.columns:
            if feat not in test_df.columns:
                continue
            shap_val = shap_df.loc[idx, feat]
            feat_value = test_df.loc[idx, feat]
            feat_avg = feature_avgs.get(feat, np.nan)
            if pd.isna(feat_avg):
                feat_avg = "N/A"
            feat_desc = get_feature_description(config_feature_desc, feat)
            if feat in list_binary:
                if feat_value == 0:
                    feat_value = "no"
                elif feat_value == 1:
                    feat_value = "yes"

            feature_rows.append(
                {
                    "feature_name": feat,
                    "feature_importance": shap_val,
                    "feature_value": feat_value,
                    "feature_average": feat_avg,
                    "feature_description": feat_desc,
                }
            )
        # Order features by descending absolute SHAP importance
        feature_rows = sorted(
            feature_rows, key=lambda x: abs(x["feature_importance"]), reverse=True
        )
        # Keep only the top N features
        feature_rows = feature_rows[:top_features]

        instance = {
            "task_description": task_description,
            "predicted_class": predicted_class,
            "dataset_description": dataset_description,
            "target_description": target_description,
            "feature_desc": feature_rows,
        }
        inputs.append(instance)
    return inputs


def explingo_prepare_inputs(ds_name, top_features=TOP_FEATURES):
    """
    Prepare LLM input instances for Explingo:
      - example_narratives (few-shot, sign-consistent; never the same feature set as the instance explanation),
      - explanation_format,
      - context,
      - explanation.
    """
    EXPLANATION_FORMAT = "(feature name, feature value, SHAP feature contribution)"
    EXAMPLE_FEATURES_PER_SIGN = 5  # up to 5 strongest same-sign features per example
    MAX_EXAMPLES = 3
    EPS = 1e-6  # treat tiny SHAPs as ~0

    # Expect in scope:
    # - prompt_configs
    # - config_data with "SHAP_RESULTS_FOLDER"
    # - get_feature_description(feature_config, feature_name) -> str
    prompt_config = prompt_configs[ds_name]

    task_description = prompt_config["task_description"]
    dataset_description = prompt_config["dataset_description"]
    target_description = prompt_config["target_description"]
    config_feature_desc = prompt_config["feature_desc"]

    base_folder = config_data["SHAP_RESULTS_FOLDER"]
    test_sample_path = os.path.join(base_folder, f"{ds_name}_test_sample.csv")
    shap_values_path = os.path.join(base_folder, f"{ds_name}_shap_values.csv")

    test_df = pd.read_csv(test_sample_path).reset_index(drop=True)
    shap_df = pd.read_csv(shap_values_path, index_col=0).reset_index(drop=True)

    # Keep only common features to avoid key errors
    common_feats = [c for c in shap_df.columns if c in test_df.columns]
    shap_df = shap_df[common_feats]
    test_df = test_df[common_feats]

    def _fmt_val(v):
        if isinstance(v, (int, float)):
            return str(int(v)) if float(v).is_integer() else f"{float(v):.2f}"
        return str(v)

    inputs = []

    for idx in test_df.index:
        # Build list of (feat_name, desc, value, shap) for this instance
        feature_data = []
        for feat in common_feats:
            feat_value = test_df.at[idx, feat]
            shap_val = shap_df.at[idx, feat]

            feat_desc = get_feature_description(
                config_feature_desc,
                feat,
                description=True,
            )
            feature_data.append((feat, feat_desc, feat_value, shap_val))

        # Sort by |SHAP| and take top-k for the explanation
        feature_data.sort(key=lambda x: abs(x[3]), reverse=True)
        top_k = feature_data[: max(1, top_features)]

        # Explanation text
        # if feature in list of binary features, change 0 to "no", 1 to "yes"
        explanation_parts = []
        for feature, desc, val, shap in top_k:
            if feature in list_binary:
                if val == 0:
                    val = "no"
                elif val == 1:
                    val = "yes"
            explanation_parts.append(
                f"({feature} ({desc}), {_fmt_val(val)}, {shap:.4f})"
            )
        explanation = ", ".join(explanation_parts)

        # For "avoid same configuration" checks
        explain_feat_names = [feat for (feat, _, _, _) in top_k]
        explain_set = set(explain_feat_names)
        explain_len = len(explain_feat_names)

        # -------- Few-shot example narratives (sign-consistent, no equal set) --------
        example_narratives = []
        seen_example_sets = set()  # dedupe examples by feature-name set

        available_indices = [i for i in shap_df.index if i != idx]
        if len(available_indices) >= 3:
            shap_sums = shap_df.loc[available_indices].abs().sum(axis=1)
            low_idx = shap_sums.nsmallest(1).index[0]
            sorted_indices = shap_sums.sort_values().index
            med_idx = sorted_indices[len(sorted_indices) // 2]
            high_idx = shap_sums.nlargest(1).index[0]

            for sample_idx in [low_idx, med_idx, high_idx]:
                # Candidates for this sample as (feat, desc, val, shap)
                candidates = []
                for feat in common_feats:
                    val = test_df.at[sample_idx, feat]
                    sv = shap_df.at[sample_idx, feat]
                    desc = get_feature_description(
                        config_feature_desc, feat, description=True
                    )
                    candidates.append((feat, desc, val, sv))

                # Split by sign and sort by |SHAP|
                pos_parts = [t for t in candidates if t[3] > EPS]
                neg_parts = [t for t in candidates if t[3] < -EPS]
                pos_parts.sort(key=lambda x: abs(x[3]), reverse=True)
                neg_parts.sort(key=lambda x: abs(x[3]), reverse=True)

                for parts, sign_label in [
                    (pos_parts, "increase"),
                    (neg_parts, "decrease"),
                ]:
                    if len(example_narratives) >= MAX_EXAMPLES:
                        break
                    if not parts:
                        continue

                    # Take up to N strongest of the same sign
                    selected = parts[:EXAMPLE_FEATURES_PER_SIGN]

                    # If selected set equals the explanation's set (same size + same members),
                    # try to swap the weakest member with the next best non-explanation feature of same sign.
                    def _feature_name_set(items):
                        return frozenset([f for (f, _, _, _) in items])

                    def _try_avoid_equal_set(selected_items, pool_items):
                        sel = list(selected_items)
                        sel_set = set([f for (f, _, _, _) in sel])
                        # Equality only possible if lengths match explain_len and sets equal
                        if len(sel) == explain_len and sel_set == explain_set:
                            # Find a replacement candidate not in explanation set
                            replacement = next(
                                (
                                    t
                                    for t in pool_items
                                    if t[0] not in explain_set and t[0] not in sel_set
                                ),
                                None,
                            )
                            if replacement is not None:
                                # Replace the weakest (min |SHAP|) element from the selection
                                weakest_idx = min(
                                    range(len(sel)), key=lambda i: abs(sel[i][3])
                                )
                                sel[weakest_idx] = replacement
                            else:
                                # If no replacement, shrink by one to break equality (keep strongest)
                                if len(sel) > 1:
                                    sel = sel[:-1]
                        return sel

                    selected = _try_avoid_equal_set(
                        selected, parts[EXAMPLE_FEATURES_PER_SIGN:]
                    )

                    # Deduplicate against already-emitted examples by feature-name set
                    name_set = _feature_name_set(selected)
                    if name_set in seen_example_sets or (
                        len(selected) == explain_len
                        and set([f for (f, _, _, _) in selected]) == explain_set
                    ):
                        # If still identical (e.g., not enough alternates), skip this example
                        continue

                    seen_example_sets.add(name_set)

                    # Build strings
                    # if feature in list of binary features, change 0 to "no", 1 to "yes"
                    for i in range(len(selected)):
                        if selected[i][0] in list_binary:
                            if selected[i][2] == 0:
                                selected[i] = (
                                    selected[i][0],
                                    selected[i][1],
                                    "no",
                                    selected[i][3],
                                )
                            elif selected[i][2] == 1:
                                selected[i] = (
                                    selected[i][0],
                                    selected[i][1],
                                    "yes",
                                    selected[i][3],
                                )
                    example_str = ", ".join(
                        f"({feat} ({desc}), {_fmt_val(val)}, {shap:.4f})"
                        for (feat, desc, val, shap) in selected
                    )

                    feature_names = [
                        f"{feat} ({desc})" for (feat, desc, _, _) in selected
                    ]
                    if len(feature_names) == 1:
                        names_txt = feature_names[0]
                    elif len(feature_names) == 2:
                        names_txt = f"{feature_names[0]} and {feature_names[1]}"
                    else:
                        names_txt = (
                            ", ".join(feature_names[:-1]) + f", and {feature_names[-1]}"
                        )

                    narrative = (
                        f"The features {names_txt} {sign_label} the predicted value."
                    )
                    example_narratives.append((example_str, narrative))

                if len(example_narratives) >= MAX_EXAMPLES:
                    break

        instance = {
            "explanation_format": EXPLANATION_FORMAT,
            "explanation": explanation,
            "context": f"{prompt_config['task_description']}",
            "example_narratives": example_narratives,
        }
        inputs.append(instance)

    return inputs


def main():
    all_inputs = {}
    methods = [
        # "xaistories",
        "explingo_narratives",
        ]

    for ds_name, _ in config_data["DATASETS_INFO"].items():
        for method in methods:
            if method == "explingo_narratives":
                inputs = explingo_prepare_inputs(ds_name)
                output_dir = os.path.join(config_data["LLM_INPUTS_FOLDER"], method)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{ds_name}_llm_inputs.json")
                with open(output_file, "w") as f:
                    json.dump(inputs, f, indent=4, default=str)
                print(f"Explingo LLM inputs for {ds_name} saved to {output_file}")

            elif method == "xaistories":
                inputs = xaistories_prepare_inputs(ds_name)
                all_inputs[ds_name] = inputs
                output_dir = os.path.join(config_data["LLM_INPUTS_FOLDER"], method)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{ds_name}_llm_inputs.json")
                with open(output_file, "w") as f:
                    json.dump(inputs, f, indent=4, default=str)
                print(f"XAIstories LLM inputs for {ds_name} saved to {output_file}")

            else:
                print(f"Method {method} not recognized.")


if __name__ == "__main__":
    main()
