# Train utils
import os
import random
import shap
import numpy as np
import pandas as pd
import yaml
import shap
import warnings


warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def load_config(path="config.yaml"):
    with open(path, "r") as file:
        config = yaml.safe_load(file)
    return config


def drop_or_fill_missing(df):
    n_rows = len(df)
    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count / n_rows >= 0.5:
            df.drop(columns=[col], inplace=True)
        else:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                mode_val = df[col].mode(dropna=True)
                if len(mode_val) > 0:
                    df[col].fillna(mode_val[0], inplace=True)
                else:
                    df[col].fillna("Unknown", inplace=True)
    return df


def encode_categorical_columns(df, cols_to_encode):
    from sklearn.preprocessing import LabelEncoder

    for col in cols_to_encode:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    return df


def encode_target(df, target_col):
    from sklearn.preprocessing import LabelEncoder

    if target_col in df.columns:
        if df[target_col].dtype == object or str(df[target_col].dtype) == "category":
            le = LabelEncoder()
            df[target_col] = le.fit_transform(df[target_col].astype(str))
    return df


def train_val_test_split_custom(X, y, test_size=0.2, min_test=30, random_state=42):
    from sklearn.model_selection import train_test_split

    total_len = len(X)
    default_test_count = int(total_len * test_size)
    test_count = (
        min_test
        if (default_test_count < min_test and total_len > min_test)
        else default_test_count
    )
    test_frac = test_count / total_len
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_frac, random_state=random_state
    )
    train_frac_rel = 0.75
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=1 - train_frac_rel, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_shap(config, best_model, X_test_sample, dataset_name):
    explainer = shap.TreeExplainer(best_model)
    shap_vals = explainer.shap_values(X_test_sample)
    if isinstance(shap_vals, list):
        final_shap = shap_vals[1]
    else:
        final_shap = shap_vals[:, :, 1]
    shap_df = pd.DataFrame(
        final_shap.round(4),
        columns=X_test_sample.columns,
        index=X_test_sample.index,
    )
    shap_path = os.path.join(
        config["SHAP_RESULTS_FOLDER"], f"{dataset_name}_shap_values.csv"
    )
    shap_df.to_csv(shap_path, index=True)
    print(f"SHAP saved to {shap_path}")
