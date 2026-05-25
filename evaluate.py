import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

from config import GESTURE_CLASSES
from train import evaluate


def measure_inference_latency(model, sample_input, device, n_runs=50):
    model.eval()
    sample_input = sample_input.to(device)

    # Warm-up
    with torch.no_grad():
        for _ in range(5):
            model(sample_input)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(sample_input)
            times.append(time.perf_counter() - t0)

    return np.mean(times) * 1000  # ms cinsinden


def get_model_size_mb(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # float32 → 4 byte
    size_mb = total * 4 / (1024 ** 2)
    return size_mb, total, trainable


def full_evaluation(model, test_loader, criterion, device, model_name,
                    sample_input=None):
    _, acc, preds, labels = evaluate(model, test_loader, criterion, device)
    f1 = f1_score(labels, preds, average="macro")
    cm = confusion_matrix(labels, preds)

    print(f"\n=== {model_name} ===")
    print(f"Top-1 Accuracy : {acc:.4f}")
    print(f"Macro F1-Score : {f1:.4f}")
    print(classification_report(labels, preds, target_names=GESTURE_CLASSES))

    # Confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=GESTURE_CLASSES, yticklabels=GESTURE_CLASSES)
    plt.title(f"Confusion Matrix — {model_name}")
    plt.ylabel("True"); plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"cm_{model_name}.png", dpi=150)
    plt.show()

    size_mb, total_params, trainable_params = get_model_size_mb(model)
    latency_ms = None
    if sample_input is not None:
        latency_ms = measure_inference_latency(model, sample_input, device)

    return {
        "model": model_name,
        "accuracy": acc,
        "f1": f1,
        "size_mb": size_mb,
        "total_params": total_params,
        "latency_ms": latency_ms,
    }


def plot_comparison(results):
    names = [r["model"] for r in results]
    accs  = [r["accuracy"] for r in results]
    f1s   = [r["f1"] for r in results]
    sizes = [r["size_mb"] for r in results]
    lats  = [r["latency_ms"] or 0 for r in results]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for ax, vals, title in zip(axes,
                               [accs, f1s, sizes, lats],
                               ["Top-1 Accuracy", "Macro F1",
                                "Model Size (MB)", "Latency (ms)"]):
        ax.bar(names, vals, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_title(title)
        ax.set_ylim(0, max(vals) * 1.2 if vals else 1)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, f"{v:.2f}",
                    ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150)
    plt.show()


def plot_training_history(histories, model_names):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for hist, name in zip(histories, model_names):
        axes[0].plot(hist["val_acc"], label=name)
        axes[1].plot(hist["val_loss"], label=name)
    axes[0].set_title("Validation Accuracy"); axes[0].legend()
    axes[1].set_title("Validation Loss");     axes[1].legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
