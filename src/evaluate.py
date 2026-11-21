import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

EMOTION_LABELS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fearful", "disgust", "surprised"
]


def evaluate_predictions(y_true, y_pred, model_name: str) -> dict:
    return {
        "Model": model_name,
        "Accuracy (%)": round(accuracy_score(y_true, y_pred) * 100, 2),
        "F1-Macro (%)": round(f1_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "Precision (%)": round(precision_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
        "Recall (%)": round(recall_score(y_true, y_pred, average="macro", zero_division=0) * 100, 2),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str = None):
    cm = confusion_matrix(y_true, y_pred)
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.divide(cm.astype("float"), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_norm * 100,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=EMOTION_LABELS,
        yticklabels=EMOTION_LABELS
    )
    plt.title(title)
    plt.ylabel("Vraie Émotion")
    plt.xlabel("Émotion Prédite")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()