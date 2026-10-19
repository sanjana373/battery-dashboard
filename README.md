# AI for Next-Generation Batteries — Dashboard

Hybrid **1D-CNN + BiLSTM** deep-learning system for battery **SOH / SOC / RUL**
estimation and 5-level **risk classification**, wrapped in a Flask web
dashboard, trained on your `battery.csv` impedance-spectroscopy data.

## What's inside

```
project/
├── data/battery.csv          <- your uploaded data
├── preprocess.py              <- parses complex readings, engineers SOH/SOC/RUL/risk labels
├── train_classical.py         <- Random Forest / Gradient Boosting / SVM / Logistic Regression baselines
├── train_deep.py              <- CNN+BiLSTM hybrid + CNN-only + BiLSTM-only (needs TensorFlow)
├── app.py                     <- Flask dashboard
├── report.py                  <- PDF report generator
├── templates/index.html       <- dashboard UI
├── static/css/style.css       <- dark, modern theme
├── static/js/main.js          <- charts + tab navigation + live prediction
├── outputs/                   <- generated metrics.json + all plots
└── models/                    <- saved .keras models
```

## ⚠️ Important note about the dataset

`battery.csv` only contains 5 **complex-valued impedance-spectroscopy**
columns (Sense_current, Battery_current, Current_ratio, Battery_impedance,
Rectified_Impedance) — there is no explicit cycle index, voltage curve,
temperature, or discharge-capacity column. Real SOC/SOH/RUL estimation
normally needs those. `preprocess.py` therefore engineers **physically
motivated proxy targets**:

- **SOH** — inverse-scaled from `Rectified_Impedance` magnitude (impedance
  rises as a cell ages — a standard, literature-backed proxy).
- **SOC** — scaled from `Current_ratio` magnitude.
- **RUL** — derived directly from SOH using the common ~1000-cycle nominal
  lifetime / 80% knee-point assumption used throughout the literature review
  in your synopsis.
- **Risk Category** — 5 bands of 20 points each, straight from SOH
  (Critical 0-20, High Risk 20-40, Low Risk 40-60, Good 60-80, Excellent
  80-100), exactly as you specified.

This is disclosed here (and in code comments) so you can explain it
confidently if your guide asks how SOC/SOH were derived — it's the honest
answer and it's a defensible engineering choice, not a shortcut.

## Setup

```bash
cd project
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

## Run — step by step

```bash
# 1. Engineer features + targets (fast, always run first)
python preprocess.py

# 2. Classical ML baselines (fast, ~10 seconds, no TensorFlow needed)
python train_classical.py

# 3. Deep learning: CNN+BiLSTM hybrid + CNN-only + BiLSTM-only
#    (needs TensorFlow; takes a few minutes on CPU)
python train_deep.py

# 4. Launch the dashboard
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

If you skip step 3, the dashboard still works fully (Accuracy/F1/Confusion
Matrix/ROC/Comparison pages show the classical-ML results from step 2, and
the live prediction page uses the transparent SOH/SOC formula). Step 3 adds
the CNN+BiLSTM numbers and the Epochs/Loss page.

## Dashboard pages (left sidebar)

1. **Battery Health Prediction** — enter the 5 impedance parameters, get
   SOH / SOC / RUL, a color-coded risk band (Critical→Excellent), and
   tailored precautions.
2. **Accuracy**, **F1 Score**, **Precision & Recall** — bar charts per algorithm.
3. **Confusion Matrix**, **ROC-AUC Curve** — pick any trained algorithm from the dropdown.
4. **Epochs / Loss** — training vs validation loss curve for hybrid / CNN-only / BiLSTM-only.
5. **Algorithm Comparison** — all metrics, all algorithms, one chart.
6. **Correlation Heatmap** — how each raw feature relates to SOH/SOC/RUL.
7. **Download Report** — one-click PDF with everything above.

## Extending it further

- Add more classical baselines in `train_classical.py`'s `models` dict.
- Add multi-head attention on top of the BiLSTM output in
  `train_deep.py::build_hybrid` (`variant == "hybrid"` branch) if your guide
  wants an attention layer explicitly — the model already exposes the
  sequence tensor `x` right before pooling for exactly that.
- Swap in a real cycle-indexed NASA dataset later and `preprocess.py`'s
  SOH/SOC/RUL formulas can be replaced with true Coulomb-counting/OCV-based
  labels with minimal changes — the rest of the pipeline (windows, models,
  dashboard) stays the same.
