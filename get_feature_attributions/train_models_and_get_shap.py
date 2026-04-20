"""
Script to train a model.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from utils.train_utils import *


def train_models_and_get_shap():
    print("\nTraining models and getting SHAP attributions\n")
    config = load_config()

    global best_model

    for dataset_name, ds_info in config["DATASETS_INFO"].items():
        print("\n" + "=" * 50)
        print(f"Processing dataset: {dataset_name}")
        print("=" * 50 + "\n")

        data_path = os.path.join(config["DATASET_FOLDER"], ds_info["filename"])
        df = pd.read_csv(data_path)
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]  # drop unnamed

        # Special case for student: create G3_binary if G3 exists
        # then rename it to "y"
        if dataset_name == "student":
            if "G3" in df.columns:
                df["G3_binary"] = df["G3"].apply(lambda x: 0 if x < 10 else 1)
            # only drop G3 if it existed
            if "G3" in df.columns:
                df.drop(columns=["G3"], inplace=True, errors="ignore")

        # If the dataset's target is not already "y", rename
        original_target = ds_info["target"]
        if original_target in df.columns and original_target != "y":
            df.rename(columns={original_target: "y"}, inplace=True)

        # Now drop dataset-specific columns
        drop_cols = ds_info["drop_cols"]
        for c in drop_cols:
            if c in df.columns:
                df.drop(columns=[c], inplace=True)

        df = drop_or_fill_missing(df)
        raw_df = df.copy()

        # encode features
        encode_categorical_columns(df, ds_info["encode_cols"])
        # encode target "y"
        if "y" not in df.columns:
            print(f"Warning: 'y' not in df.columns for {dataset_name}, skipping.")
            continue
        encode_target(df, "y")

        print(f"After preprocessing: {df.shape[0]} rows, {df.shape[1]} columns.")

        # separate X, y
        if "y" not in df.columns:
            print(f"No 'y' in columns after rename, skipping {dataset_name}")
            continue

        X = df.drop(columns=["y"])
        y = df["y"]
        X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split_custom(
            X, y, test_size=0.2, min_test=30, random_state=SEED
        )
        print(f"Train set: {X_train.shape[0]}")
        print(f"Val set: {X_val.shape[0]}")
        print(f"Test set: {X_test.shape[0]}")

        X_train_val = np.vstack([X_train, X_val])
        y_train_val = np.concatenate([y_train, y_val])

        # train a model
        if dataset_name in config["FIXED_HP"]:
            hp = config["FIXED_HP"][dataset_name]
            best_model = RandomForestClassifier(
                random_state=SEED, class_weight="balanced", **hp
            )
            best_model.fit(X_train_val, y_train_val)
            print(f"Using fixed HP for {dataset_name}: {hp}")
        else:
            print(f"No HP found for {dataset_name}, skipping.")
            continue

        # evaluate
        test_probs = best_model.predict_proba(X_test)
        n_classes = len(np.unique(y_train_val))
        if n_classes == 2:
            auc_score = roc_auc_score(y_test, test_probs[:, 1])
        else:
            auc_score = roc_auc_score(y_test, test_probs, multi_class="ovr")
        print(f"Test AUC: {auc_score:.3f}")

        # sample test
        n_test = X_test.shape[0]
        sample_size = min(30, n_test)
        idx_sample = np.random.choice(X_test.index, size=sample_size, replace=False)
        X_test_sample = X_test.loc[idx_sample]

        # predictions
        preds = best_model.predict(X_test_sample)
        mapped_preds = [
            config["TARGET_MAPPINGS"][dataset_name].get(int(p), str(p)) for p in preds
        ]

        # raw test sample
        raw_test_sample = raw_df.loc[X_test_sample.index].copy()
        raw_test_sample["predicted_y"] = mapped_preds
        sample_file = os.path.join(
            config["SHAP_RESULTS_FOLDER"], f"{dataset_name}_test_sample.csv"
        )
        raw_test_sample.drop(columns=["y"], inplace=True)
        raw_test_sample.to_csv(sample_file, index=False)
        print(f"Saved test sample to {sample_file}")

        # shap
        calculate_shap(config, best_model, X_test_sample, dataset_name)
