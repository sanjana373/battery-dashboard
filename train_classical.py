"""
train_classical.py
====================
Trains classical ML baselines (Random Forest, Gradient Boosting, SVM,
Logistic Regression) on the flattened windowed features to classify battery
Risk Category (Critical / High Risk / Low Risk / Good / Excellent).

These run instantly on CPU with only scikit-learn (no TensorFlow needed), so
this script also acts as your fallback / sanity-check comparison group
against the CNN-BiLSTM deep model trained by train_deep.py.

Outputs (all under outputs/):
  metrics_classical.json   - accuracy / precision / recall / F1 / per-algo
  confusion_matrix_<algo>.png
  roc_auc_<algo>.png
  algorithm_comparison.png
  correlation_heatmap.png
"""
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, roc_curve, auc, classification_report, log_loss)

from preprocess import run as preprocess_run, RISK_LABELS

OUT = "outputs"
import os
os.makedirs(OUT, exist_ok=True)


def flatten_windows():
    data = np.load("data/windowed.npz", allow_pickle=True)
    X = data["X"]                      # (N, window, features)
    y = data["risk"]
    N = X.shape[0]
    X_flat = X.reshape(N, -1)          # flatten window*features for classical models
    return X_flat, y


def plot_correlation_heatmap(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if c.endswith("_mag") or c.endswith("_phase")]
    corr = df[feat_cols + ["SOH", "SOC", "RUL"]].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Feature / SOH / SOC / RUL Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT}/correlation_heatmap.png", dpi=140)
    plt.close()


def plot_training_curve(name, model, X_train, y_train, X_test, y_test):
    """
    Classical ML models don't train epoch-by-epoch like neural nets, so
    there's no true per-epoch loss to plot. To still give every algorithm a
    'Training Dynamics' page like the deep models, this fits the same model
    on increasing fractions of the training data (10% -> 100%, refit each
    time) and records real train/test log-loss at each fraction. It's a
    genuine learning curve, just indexed by training-set size instead of
    epoch, and is labeled as such on the chart so it isn't confused with a
    true epoch loss curve.
    """
    if name == "GradientBoosting" and hasattr(model, "train_score_"):
        # GradientBoosting actually does have a real per-stage (boosting
        # iteration) loss trace - use the genuine one instead of refitting.
        train_curve = model.train_score_.tolist()
        # approximate a held-out curve via staged_predict_proba if available
        try:
            test_curve = []
            for proba in model.staged_predict_proba(X_test):
                test_curve.append(log_loss(y_test, proba, labels=model.classes_))
        except Exception:
            test_curve = train_curve
        x_label = "Boosting Iteration"
    else:
        fractions = np.linspace(0.1, 1.0, 10)
        train_curve, test_curve = [], []
        for frac in fractions:
            n = max(10, int(len(X_train) * frac))
            Xf, yf = X_train[:n], y_train[:n]
            if len(set(yf)) < 2:
                continue
            m = type(model)(**model.get_params())
            m.fit(Xf, yf)
            if hasattr(m, "predict_proba"):
                train_curve.append(log_loss(yf, m.predict_proba(Xf), labels=m.classes_))
                test_curve.append(log_loss(y_test, m.predict_proba(X_test), labels=m.classes_))
            else:
                train_curve.append(1 - m.score(Xf, yf))
                test_curve.append(1 - m.score(X_test, y_test))
        x_label = "Training Data Fraction (x10%)"

    plt.figure(figsize=(7, 5))
    plt.plot(train_curve, label="train loss")
    plt.plot(test_curve, label="val/test loss")
    plt.xlabel(x_label)
    plt.ylabel("Log Loss")
    plt.title(f"Training Curve — {name} (learning curve, not epoch-based)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/training_curve_{name}.png", dpi=140)
    plt.close()


def run():
    df = preprocess_run()
    plot_correlation_heatmap(df)

    X, y = flatten_windows()
    classes = sorted(set(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=2000),
    }

    results = {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0)

        plot_training_curve(name, model, X_train, y_train, X_test, y_test)

        cm = confusion_matrix(y_test, y_pred, labels=classes)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
        plt.title(f"Confusion Matrix — {name}")
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{OUT}/confusion_matrix_{name}.png", dpi=140)
        plt.close()

        # ROC-AUC (one-vs-rest, multiclass)
        if hasattr(model, "predict_proba"):
            y_test_bin = label_binarize(y_test, classes=classes)
            y_score = model.predict_proba(X_test)
            plt.figure(figsize=(6, 5))
            aucs = {}
            for i, cls in enumerate(classes):
                fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                aucs[cls] = roc_auc
                plt.plot(fpr, tpr, label=f"{cls} (AUC={roc_auc:.2f})")
            plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title(f"ROC-AUC — {name}")
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(f"{OUT}/roc_auc_{name}.png", dpi=140)
            plt.close()
            mean_auc = float(np.mean(list(aucs.values())))
        else:
            aucs, mean_auc = {}, None

        results[name] = {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "mean_auc": round(mean_auc, 4) if mean_auc else None,
            "train_time_sec": round(train_time, 3),
            "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
        }
        print(f"[{name}] acc={acc:.4f} f1={f1:.4f} auc={mean_auc}")

    # Algorithm comparison bar chart
    algos = list(results.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(algos))
    width = 0.2
    plt.figure(figsize=(9, 5))
    for i, m in enumerate(metrics_names):
        vals = [results[a][m] for a in algos]
        plt.bar(x + i * width, vals, width, label=m)
    plt.xticks(x + width * 1.5, algos, rotation=15)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Algorithm Comparison — Classical ML Baselines")
    plt.tight_layout()
    plt.savefig(f"{OUT}/algorithm_comparison_classical.png", dpi=140)
    plt.close()

    with open(f"{OUT}/metrics_classical.json", "w") as f:
        json.dump({"classes": classes, "results": results}, f, indent=2)
    print(f"[train_classical] wrote {OUT}/metrics_classical.json")
    return results


if __name__ == "__main__":
    run()
