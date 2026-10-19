"""
train_deep.py
==============
Trains the main hybrid 1D-CNN + BiLSTM model (plus standalone CNN-only and
BiLSTM-only baselines) on the windowed data produced by preprocess.py.

The model is MULTI-OUTPUT:
  - risk_output : softmax over 5 classes (Critical..Excellent)   -> classification
  - soh_output  : linear                                          -> regression
  - soc_output  : linear                                          -> regression
  - rul_output  : linear (relu-clamped)                           -> regression

Requires TensorFlow (pip install tensorflow). Run AFTER preprocess.py has
produced data/windowed.npz (train_classical.py does this automatically too,
so running that first is fine).

Outputs (outputs/):
  metrics_deep.json                 - accuracy/F1/precision/recall/loss per model
  training_curve_<model>.png        - loss vs epoch (train & val)
  confusion_matrix_deep_<model>.png
  roc_auc_deep_<model>.png
  algorithm_comparison_full.png     - classical + deep side by side
models/ (saved trained models + scaler info) for the Flask app to load.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, roc_curve, auc)

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from preprocess import RISK_LABELS

OUT = "outputs"
MODEL_DIR = "models"
os.makedirs(OUT, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def load_data():
    data = np.load("data/windowed.npz", allow_pickle=True)
    X = data["X"].astype("float32")
    soh, soc, rul = data["soh"].astype("float32"), data["soc"].astype("float32"), data["rul"].astype("float32")
    risk = data["risk"]
    feat_cols = data["feat_cols"]
    mu, sigma = data["mu"], data["sigma"]

    le = LabelEncoder()
    le.fit(RISK_LABELS)
    risk_enc = le.transform(risk)

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=risk_enc)

    return (X[train_idx], X[test_idx],
            soh[train_idx], soh[test_idx],
            soc[train_idx], soc[test_idx],
            rul[train_idx], rul[test_idx],
            risk_enc[train_idx], risk_enc[test_idx],
            le, feat_cols, mu, sigma)


def build_hybrid(input_shape, n_classes, variant="hybrid"):
    """variant: 'hybrid' (CNN+BiLSTM), 'cnn_only', 'bilstm_only'."""
    inp = layers.Input(shape=input_shape, name="sequence_input")
    x = inp

    if variant in ("hybrid", "cnn_only"):
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
        x = layers.Dropout(0.2)(x)

    if variant in ("hybrid", "bilstm_only"):
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(64))(x)
    else:  # cnn_only: pool down to a flat vector
        x = layers.GlobalAveragePooling1D()(x)

    if variant == "hybrid":
        # Multi-head self-attention over the CNN+BiLSTM feature sequence would
        # need return_sequences=True on the last BiLSTM; keep it simple/stable:
        pass

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    risk_out = layers.Dense(n_classes, activation="softmax", name="risk_output")(x)
    soh_out = layers.Dense(1, activation="linear", name="soh_output")(x)
    soc_out = layers.Dense(1, activation="linear", name="soc_output")(x)
    rul_out = layers.Dense(1, activation="relu", name="rul_output")(x)

    model = models.Model(inp, [risk_out, soh_out, soc_out, rul_out], name=f"battery_{variant}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss={
            "risk_output": "sparse_categorical_crossentropy",
            "soh_output": "mse",
            "soc_output": "mse",
            "rul_output": "mse",
        },
        loss_weights={"risk_output": 1.0, "soh_output": 0.01, "soc_output": 0.01, "rul_output": 0.0005},
        metrics={"risk_output": "accuracy"},
    )
    return model


def train_one(variant, X_train, X_test, soh_train, soh_test, soc_train, soc_test,
              rul_train, rul_test, risk_train, risk_test, n_classes):
    model = build_hybrid(X_train.shape[1:], n_classes, variant)

    es = callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    hist = model.fit(
        X_train,
        {"risk_output": risk_train, "soh_output": soh_train, "soc_output": soc_train, "rul_output": rul_train},
        validation_split=0.15,
        epochs=120,
        batch_size=32,
        callbacks=[es],
        verbose=2,
    )

    # ---- plot loss curve ----
    plt.figure(figsize=(7, 5))
    plt.plot(hist.history["loss"], label="train loss")
    plt.plot(hist.history["val_loss"], label="val loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title(f"Training Curve — {variant}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/training_curve_{variant}.png", dpi=140)
    plt.close()

    # ---- evaluate ----
    preds = model.predict(X_test, verbose=0)
    risk_prob, soh_pred, soc_pred, rul_pred = preds
    risk_pred = np.argmax(risk_prob, axis=1)

    acc = accuracy_score(risk_test, risk_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(risk_test, risk_pred, average="weighted", zero_division=0)

    cm = confusion_matrix(risk_test, risk_pred, labels=list(range(n_classes)))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=RISK_LABELS, yticklabels=RISK_LABELS)
    plt.title(f"Confusion Matrix — {variant}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUT}/confusion_matrix_deep_{variant}.png", dpi=140)
    plt.close()

    y_test_bin = label_binarize(risk_test, classes=list(range(n_classes)))
    plt.figure(figsize=(6, 5))
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], risk_prob[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, label=f"{RISK_LABELS[i]} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.title(f"ROC-AUC — {variant}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/roc_auc_deep_{variant}.png", dpi=140)
    plt.close()

    soh_mae = float(np.mean(np.abs(soh_pred.flatten() - soh_test)))
    soc_mae = float(np.mean(np.abs(soc_pred.flatten() - soc_test)))
    rul_mae = float(np.mean(np.abs(rul_pred.flatten() - rul_test)))

    model.save(f"{MODEL_DIR}/{variant}.keras")

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "mean_auc": round(float(np.mean(aucs)), 4),
        "soh_mae": round(soh_mae, 3),
        "soc_mae": round(soc_mae, 3),
        "rul_mae": round(rul_mae, 3),
        "final_train_loss": round(float(hist.history["loss"][-1]), 5),
        "final_val_loss": round(float(hist.history["val_loss"][-1]), 5),
        "epochs_run": len(hist.history["loss"]),
        "loss_curve": [round(float(v), 5) for v in hist.history["loss"]],
        "val_loss_curve": [round(float(v), 5) for v in hist.history["val_loss"]],
    }


def run():
    (X_train, X_test, soh_train, soh_test, soc_train, soc_test, rul_train, rul_test,
     risk_train, risk_test, le, feat_cols, mu, sigma) = load_data()
    n_classes = len(le.classes_)

    results = {}
    for variant in ["hybrid", "cnn_only", "bilstm_only"]:
        print(f"\n=== Training {variant} ===")
        results[variant] = train_one(
            variant, X_train, X_test, soh_train, soh_test, soc_train, soc_test,
            rul_train, rul_test, risk_train, risk_test, n_classes)
        print(results[variant])

    # merge with classical results if present, for the full comparison chart
    combined = dict(results)
    if os.path.exists(f"{OUT}/metrics_classical.json"):
        with open(f"{OUT}/metrics_classical.json") as f:
            classical = json.load(f)["results"]
        combined.update(classical)

    algos = list(combined.keys())
    metrics_names = ["accuracy", "precision", "recall", "f1_score"]
    x = np.arange(len(algos))
    width = 0.2
    plt.figure(figsize=(11, 5))
    for i, m in enumerate(metrics_names):
        vals = [combined[a].get(m, 0) for a in algos]
        plt.bar(x + i * width, vals, width, label=m)
    plt.xticks(x + width * 1.5, algos, rotation=20)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Algorithm Comparison — All Models")
    plt.tight_layout()
    plt.savefig(f"{OUT}/algorithm_comparison_full.png", dpi=140)
    plt.close()

    with open(f"{OUT}/metrics_deep.json", "w") as f:
        json.dump({"classes": RISK_LABELS, "results": results}, f, indent=2)

    # save encoder classes + normalisation stats for the Flask app
    np.savez(f"{MODEL_DIR}/norm_stats.npz", mu=mu, sigma=sigma, feat_cols=feat_cols,
             classes=np.array(RISK_LABELS))
    print(f"\n[train_deep] wrote {OUT}/metrics_deep.json and saved models/*.keras")


if __name__ == "__main__":
    run()
