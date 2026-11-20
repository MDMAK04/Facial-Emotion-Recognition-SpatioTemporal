import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

CLASS_NAMES = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """Calcule Accuracy, Precision, Recall, F1-score (Macro & Weighted) et UAR."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "Model": model_name,
        "Accuracy (%)": round(acc * 100, 2),
        "Macro Precision (%)": round(prec_macro * 100, 2),
        "Macro Recall / UAR (%)": round(rec_macro * 100, 2),
        "Macro F1-Score (%)": round(f1_macro * 100, 2),
        "Weighted F1-Score (%)": round(f1_weighted * 100, 2)
    }
    return metrics

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str = None):
    """Affiche et sauvegarde la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES
    )
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def measure_inference_time(model: torch.nn.Module, sample_input: torch.Tensor, device: torch.device, runs: int = 100) -> float:
    """Mesure le temps moyen d'inférence par vidéo (en millisecondes)."""
    model.eval()
    sample_input = sample_input.to(device)
    with torch.no_grad():
        for _ in range(10):  # Warm-up GPU
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(runs):
            _ = model(sample_input)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.time() - start) / runs
    return round(elapsed * 1000, 3)