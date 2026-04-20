"""
Main script to train ML models and get SHAP attributions.
"""

from get_feature_attributions.train_models_and_get_shap import train_models_and_get_shap


def main():
    # Train ML models and get feature attribution with SHAP
    train_models_and_get_shap()


if __name__ == "__main__":
    main()
