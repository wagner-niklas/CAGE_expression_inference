import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    classification_report,
)
import os
import subprocess

ONLY_INFERENCE = False
root_dir = "."
command = "echo test"
command = "python3 generate_csv.py"

label_mapping = {
    "Neutral": 0,
    "Happy": 1,
    "Sad": 2,
    "Surprise": 3,
    "Fear": 4,
    "Disgust": 5,
    "Anger": 6,
    "Contempt": 7,
}


def get_subdirectories(directory):
    subdirs = []
    for item in os.listdir(directory):
        full_path = os.path.abspath(os.path.join(directory, item))
        if os.path.isdir(full_path):
            subdirs.append(full_path)
    return subdirs


def get_files_in_directory(directory):
    files = []
    # Iterate over each item in the directory
    for item in os.listdir(directory):
        # Check if it's a file
        if os.path.isfile(os.path.join(directory, item)):
            files.append(item)
    return files


def concordance_correlation_coefficient(true_values, pred_values):
    mean_true = np.mean(true_values)
    mean_pred = np.mean(pred_values)

    num = 2 * np.cov(true_values, pred_values)[0, 1]
    den = np.var(true_values) + np.var(pred_values) + (mean_true - mean_pred) ** 2
    return num / den


def print_discrete(true_labels, pred_labels):
    if max(true_labels) == 7:
        class_names = [
            "Anger",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
            "Contempt",
        ]
    else:
        class_names = [
            "Anger",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
        ]

    mapped_labels = [label_mapping[name] for name in class_names]
    map = classification_report(
        true_labels,
        pred_labels,
        labels=mapped_labels,
        target_names=class_names,
        zero_division=0.0,
        digits=3,
        output_dict=True,
    )
    precision = map["weighted avg"]["precision"]
    recall = map["weighted avg"]["recall"]
    f1 = map["weighted avg"]["f1-score"]
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")


def evaluate(path: str):
    df = pd.read_csv(path)
    discrete = "cat_pred" in df.columns
    va = "val_pred" in df.columns
    if va:
        true_values = list(df["val_true"]) + list(df["aro_true"])
        pred_values = list(df["val_pred"]) + list(df["aro_pred"])
    if va:
        mse = mean_squared_error(true_values, pred_values)
        mae = mean_absolute_error(true_values, pred_values)
        rmse = root_mean_squared_error(true_values, pred_values)
        ccc = concordance_correlation_coefficient(true_values, pred_values)
    print(path)
    if discrete:
        print_discrete(df["cat_true"], df["cat_pred"])
    if va:
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Concordance Correlation Coefficient (CCC): {ccc:.4f}")


for subdir in get_subdirectories("."):
    files = get_files_in_directory(subdir)
    if ONLY_INFERENCE is False:
        if "model.pt" in files:
            result = subprocess.run(
                command, shell=True, cwd=subdir, capture_output=True, text=True
            )
    files = get_files_in_directory(subdir)
    if "inference.csv" in files:
        evaluate(os.path.join(subdir, "inference.csv"))
        print("\n")
        print(50 * "-")
        print("\n")
