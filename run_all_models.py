import pandas as pd
from model_pipeline import evaluate_models_with_kfold
import os, json
from datetime import datetime

if __name__ == "__main__":
    config = {
        "data_path": "data/final_xrf_metal_data.csv",
        "label_column": "CityGroup",  # 'Year', 'PeriodGroup', 'CityGroup', or 'TopArtist'
        "xrf_columns": [f"ch_{i:04d}" for i in range(1, 2049)],
        "metal_columns": ['Fe', 'Ni', 'Cu', 'Zn', 'Ag', 'Cd', 'Sn', 'Sb', 'Au', 'Hg', 'Pb', 'Bi'],
        "task": "classification",  # "classification" or "regression"
        "n_splits": 5,  # K in KFold
        "latent_dim": 32, # for autoencoder
        "num_epochs": 200, # for autoencoder
        "random_state": 42, # random seeds you like
        "device": "cpu",
        "preprocessing": "log",  # "none", "sqrt", or "log" (for autoencoder and CNN)
        "use_optuna": False, # hyperparameter tuning framework for tree-based models
        "output_confusion_matrix": True,
        "conf_matrix_condition": {
            "method": "metal_only",
            "model": "LightGBM"
        },

        # choose the input and downstreams you want to run
        "input_methods": [
            "metal_only",
            "latent_only",
            "feature",
            # "xrf_only",
            # "metal+PLS",
            # "xrf+CNN"
        ],
        "models": [
            "RandomForest",
            "XGBoost",
            "LightGBM",
            # "PLSRegression",
            # "CNN"
        ]
    }

    df_results = evaluate_models_with_kfold(**config)

    if config["task"] == "classification":
        df_output = df_results.groupby(['method', 'model'])[['acc', 'f1', 'recall']].mean()
    else:
        df_output = df_results.groupby(['method', 'model'])[['r2', 'rmse', 'mae']].mean()
    print(df_output)

    config_to_save = {
        k: v for k, v in config.items()
        if k not in ["data_path", "xrf_columns", "metal_columns", "label_column"]
    }
    os.makedirs("log", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_path = f"log/metrics/log_{timestamp}.csv"
    df_doc = df_output.reset_index()
    df_doc.to_csv(metrics_path, index=False)
    config_path = f"log/config/config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(config_to_save, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    print(f"Config saved to: {config_path}")


    # Confusion matrix visualization (optional)
    if config.get("output_confusion_matrix", False):
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        from sklearn.preprocessing import LabelEncoder
        import matplotlib.pyplot as plt

        if hasattr(evaluate_models_with_kfold, 'all_y_true') and hasattr(evaluate_models_with_kfold, 'all_y_pred'):
            all_y_true = evaluate_models_with_kfold.all_y_true
            all_y_pred = evaluate_models_with_kfold.all_y_pred
        else:
            from model_pipeline import all_y_true, all_y_pred

        if all_y_true and all_y_pred:
            df = pd.read_csv(config["data_path"])
            label_encoder = LabelEncoder()
            label_encoder.fit(df[config["label_column"]])
            class_names = label_encoder.classes_

            cm = confusion_matrix(all_y_true, all_y_pred, labels=range(len(class_names)))
            fig, ax = plt.subplots(figsize=(14, 14))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(ax=ax, xticks_rotation=90, cmap='Blues', colorbar=False)
            for text in ax.texts:
                text.set_fontsize(20)
            plt.title("Confusion Matrix ({} on {})".format(
                config["conf_matrix_condition"].get("model"),
                config["conf_matrix_condition"].get("method")
            ))
            plt.tight_layout()
            plt.show()
