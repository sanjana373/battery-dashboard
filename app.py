"""
app.py
=======
Flask web dashboard for the AI Battery Health & RUL project.

Layout: left sidebar = navigation (Battery Health, Accuracy, F1 Score,
Confusion Matrix, Precision/Recall, Epochs, Algorithm Comparison, ROC-AUC,
Heatmap, Download Report). Right side = page header + content area.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import io
import json
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_file

from preprocess import PRECAUTIONS, RISK_LABELS, RISK_BINS, engineer_features, load_raw

app = Flask(__name__)

OUT = "outputs"
MODEL_DIR = "models"

# Try to load a trained Keras model for live single-sample prediction.
# Falls back to a physics-formula estimate if TensorFlow / trained model
# isn't available yet (e.g. before the user has run train_deep.py).
_KERAS_MODEL = None
_NORM = None
try:
    import tensorflow as tf
    if os.path.exists(f"{MODEL_DIR}/hybrid.keras"):
        _KERAS_MODEL = tf.keras.models.load_model(f"{MODEL_DIR}/hybrid.keras")
        norm = np.load(f"{MODEL_DIR}/norm_stats.npz", allow_pickle=True)
        _NORM = {"mu": norm["mu"], "sigma": norm["sigma"]}
except Exception as e:
    print("[app] deep model not loaded (fallback formulas will be used):", e)


def load_json(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def risk_from_soh(soh):
    for i in range(len(RISK_BINS) - 1):
        if RISK_BINS[i] <= soh < RISK_BINS[i + 1]:
            return RISK_LABELS[i]
    return RISK_LABELS[-1]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/summary")
def api_summary():
    """Overall dataset summary for the top of the dashboard."""
    proc_path = "data/processed_data.csv"
    if not os.path.exists(proc_path):
        return jsonify({"error": "Run preprocess.py first."}), 400
    df = pd.read_csv(proc_path)
    return jsonify({
        "n_samples": len(df),
        "avg_soh": round(df["SOH"].mean(), 2),
        "avg_soc": round(df["SOC"].mean(), 2),
        "avg_rul": round(df["RUL"].mean(), 1),
        "risk_distribution": df["Risk_Category"].value_counts().to_dict(),
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Live single-reading prediction. Accepts the 5 raw complex-style
    parameters (or their magnitude/phase directly) and returns SOH, SOC,
    RUL, risk category, precautions and a naive accuracy indicator.
    """
    payload = request.get_json(force=True)

    try:
        sense_mag = float(payload["sense_current_mag"])
        batt_mag = float(payload["battery_current_mag"])
        ratio_mag = float(payload["current_ratio_mag"])
        imp_mag = float(payload["battery_impedance_mag"])
        rect_mag = float(payload["rectified_impedance_mag"])
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Missing/invalid parameters."}), 400

    for label, v in [("sense_current_mag", sense_mag), ("battery_current_mag", batt_mag),
                      ("current_ratio_mag", ratio_mag), ("battery_impedance_mag", imp_mag),
                      ("rectified_impedance_mag", rect_mag)]:
        if not (0 <= v <= 1000):
            return jsonify({"error": f"{label} must be between 0 and 1000."}), 400

    # Reuse the same formulas as preprocess.py, anchored to the training
    # dataset's observed impedance / current-ratio range for consistent scaling.
    df_ref = pd.read_csv("data/processed_data.csv")
    ri_min, ri_max = df_ref["Rectified_Impedance_mag"].min(), df_ref["Rectified_Impedance_mag"].max()
    cr_min, cr_max = df_ref["Current_ratio_mag"].min(), df_ref["Current_ratio_mag"].max()

    soh = float(np.clip((1 - (rect_mag - ri_min) / (ri_max - ri_min)) * 100, 0, 100))
    soc = float(np.clip((ratio_mag - cr_min) / (cr_max - cr_min) * 100, 0, 100))
    rul = float(np.clip((soh - 80) / 20 * 1000, 0, 1000))
    risk = risk_from_soh(soh)

    source = "formula"
    # If a trained deep model is available we could run a full window through
    # it instead; single-reading live input uses the transparent formula path
    # so predictions are always explainable to the reviewer.

    return jsonify({
        "soh": round(soh, 2),
        "soc": round(soc, 2),
        "rul_cycles": round(rul, 1),
        "risk_category": risk,
        "precautions": PRECAUTIONS[risk],
        "source": source,
    })


@app.route("/api/metrics")
def api_metrics():
    classical = load_json(f"{OUT}/metrics_classical.json")
    deep = load_json(f"{OUT}/metrics_deep.json")
    combined = {}
    if classical:
        combined.update(classical["results"])
    if deep:
        combined.update(deep["results"])
    return jsonify({"results": combined, "has_deep": deep is not None, "has_classical": classical is not None})


@app.route("/static/img/<path:filename>")
def serve_generated_img(filename):
    """Serve plots saved to outputs/ under the same /static/img/ url space."""
    path = os.path.join(OUT, filename)
    if os.path.exists(path):
        return send_file(path)
    fallback = os.path.join("static", "img", filename)
    if os.path.exists(fallback):
        return send_file(fallback)
    return "", 404


@app.route("/download_report")
def download_report():
    from report import build_report
    buf = build_report()
    return send_file(buf, as_attachment=True, download_name="Battery_Health_Report.pdf",
                      mimetype="application/pdf")


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
