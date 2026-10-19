"""
report.py
==========
Builds a downloadable PDF report covering objectives, dataset summary,
engineered targets, and every metric/plot produced by train_classical.py
and train_deep.py. Used by the /download_report route in app.py, and can
also be run standalone: `python report.py` writes outputs/Battery_Health_Report.pdf
"""
import io
import json
import os
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, PageBreak)

OUT = "outputs"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="H1c", fontSize=20, leading=24, spaceAfter=14, textColor=colors.HexColor("#0b0f19")))
styles.add(ParagraphStyle(name="H2c", fontSize=14, leading=18, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#1e3a8a")))
styles.add(ParagraphStyle(name="Body", fontSize=10, leading=14))


def _img(path, width=15 * cm):
    if os.path.exists(path):
        return Image(path, width=width, height=width * 0.66)
    return None


def build_report():
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    story = []

    story.append(Paragraph("AI for Next-Generation Batteries — Results Report", styles["H1c"]))
    story.append(Paragraph(
        "Hybrid 1D-CNN + BiLSTM model for battery SOC / SOH / RUL estimation and "
        "risk classification, generated from the impedance-spectroscopy dataset "
        "(battery.csv).", styles["Body"]))

    story.append(Paragraph("Objectives", styles["H2c"]))
    objs = [
        "Extract and pre-process battery aging data (impedance, current-ratio parameters).",
        "Develop a deep learning model capturing spatial (CNN) and temporal (BiLSTM) features.",
        "Estimate State of Health (SOH) and State of Charge (SOC) in real time.",
        "Predict Remaining Useful Life (RUL) and classify risk into 5 bands.",
        "Provide precautionary guidance per risk band for battery management.",
        "Compare the hybrid model against CNN-only, BiLSTM-only and classical ML baselines.",
    ]
    for o in objs:
        story.append(Paragraph(f"&bull; {o}", styles["Body"]))

    # ---- dataset summary ----
    if os.path.exists("data/processed_data.csv"):
        df = pd.read_csv("data/processed_data.csv")
        story.append(Paragraph("Dataset Summary", styles["H2c"]))
        rows = [
            ["Samples", str(len(df))],
            ["Average SOH", f"{df['SOH'].mean():.2f}%"],
            ["Average SOC", f"{df['SOC'].mean():.2f}%"],
            ["Average RUL", f"{df['RUL'].mean():.1f} cycles"],
        ]
        t = Table(rows, colWidths=[6 * cm, 6 * cm])
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e0f2fe")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t)

        story.append(Paragraph("Risk Category Distribution", styles["H2c"]))
        dist = df["Risk_Category"].value_counts()
        rows2 = [[k, str(v)] for k, v in dist.items()]
        t2 = Table([["Risk Category", "Samples"]] + rows2, colWidths=[6 * cm, 6 * cm])
        t2.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#c7d2fe")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ]))
        story.append(t2)

    # ---- metrics tables ----
    combined = {}
    for fname in ["metrics_classical.json", "metrics_deep.json"]:
        p = f"{OUT}/{fname}"
        if os.path.exists(p):
            with open(p) as f:
                combined.update(json.load(f)["results"])

    if combined:
        story.append(Paragraph("Algorithm Performance Summary", styles["H2c"]))
        header = ["Algorithm", "Accuracy", "Precision", "Recall", "F1", "Mean AUC"]
        rows = [header]
        for name, m in combined.items():
            rows.append([name, m.get("accuracy", "-"), m.get("precision", "-"),
                         m.get("recall", "-"), m.get("f1_score", "-"), m.get("mean_auc", "-")])
        t = Table(rows, colWidths=[4 * cm] + [2.4 * cm] * 5)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b0f19")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(t)

    story.append(PageBreak())
    story.append(Paragraph("Visual Results", styles["H1c"]))

    plots = [
        ("Correlation Heatmap", f"{OUT}/correlation_heatmap.png"),
        ("Algorithm Comparison (All Models)", f"{OUT}/algorithm_comparison_full.png"),
        ("Algorithm Comparison (Classical Baselines)", f"{OUT}/algorithm_comparison_classical.png"),
    ]
    for title, path in plots:
        img = _img(path)
        if img:
            story.append(Paragraph(title, styles["H2c"]))
            story.append(img)
            story.append(Spacer(1, 10))

    # Per-algorithm section: training curve, confusion matrix, ROC-AUC for
    # EVERY algorithm in `combined` (not just a hardcoded subset), so the
    # report always covers all trained models (classical + deep).
    deep_variant_by_display = {"CNN+BiLSTM (Hybrid)": "hybrid", "1D-CNN": "cnn_only", "BiLSTM": "bilstm_only"}
    for algo_name in combined.keys():
        story.append(PageBreak())
        story.append(Paragraph(f"Algorithm: {algo_name}", styles["H1c"]))

        if algo_name in deep_variant_by_display:
            variant = deep_variant_by_display[algo_name]
            train_curve_path = f"{OUT}/training_curve_{variant}.png"
            cm_path = f"{OUT}/confusion_matrix_deep_{variant}.png"
            roc_path = f"{OUT}/roc_auc_deep_{variant}.png"
        else:
            train_curve_path = f"{OUT}/training_curve_{algo_name}.png"
            cm_path = f"{OUT}/confusion_matrix_{algo_name}.png"
            roc_path = f"{OUT}/roc_auc_{algo_name}.png"

        for title, path in [
            (f"Training Curve — {algo_name}", train_curve_path),
            (f"Confusion Matrix — {algo_name}", cm_path),
            (f"ROC-AUC — {algo_name}", roc_path),
        ]:
            img = _img(path)
            if img:
                story.append(Paragraph(title, styles["H2c"]))
                story.append(img)
                story.append(Spacer(1, 10))

    doc.build(story)
    buf.seek(0)
    return buf


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    buf = build_report()
    with open(f"{OUT}/Battery_Health_Report.pdf", "wb") as f:
        f.write(buf.read())
    print(f"[report] wrote {OUT}/Battery_Health_Report.pdf")
