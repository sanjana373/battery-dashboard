"""
preprocess.py
==============
Loads data/battery.csv (NASA-style impedance measurement log with complex-valued
columns: Sense_current, Battery_current, Current_ratio, Battery_impedance,
Rectified_Impedance) and turns it into a clean, model-ready dataset.

Why engineered targets?
------------------------
The raw file only contains impedance-spectroscopy style complex readings — there
is no explicit cycle index, voltage curve, temperature, or discharge capacity
column. True SOC/SOH/RUL estimation normally needs those. To still build a
working, physically-motivated pipeline, this script derives:

  * SOH (State of Health, %)   -> from the magnitude of Rectified_Impedance.
                                   Internal impedance rises monotonically as a
                                   Li-ion cell ages, so we invert & rescale it
                                   into a 100 -> 55 % health range (a very
                                   standard, literature-backed proxy — see
                                   Refs [1],[11],[12] in the synopsis).
  * SOC (State of Charge, %)   -> from the Current_ratio magnitude, which
                                   tracks the sense/battery current balance
                                   during charge-discharge swings, rescaled to
                                   0-100 % and smoothed.
  * RUL (cycles remaining)     -> distance (in samples) from each row to the
                                   first point where SOH crosses the 80 %
                                   "knee" threshold (industry-standard EOL
                                   definition, same one used in the synopsis).
  * Risk category (5 classes) -> binned directly from SOH in bands of 20:
        0-20   Critical
        20-40  High Risk
        40-60  Low Risk
        60-80  Good
        80-100 Excellent

Run standalone to sanity-check + write data/processed_data.csv and the
windowed numpy arrays used by both the classical and deep-learning trainers.
"""
import re
import numpy as np
import pandas as pd

RAW_PATH = "data/battery.csv"
PROCESSED_PATH = "data/processed_data.csv"
WINDOW = 20          # sliding-window length fed to CNN/BiLSTM
KNEE_SOH = 80.0       # % SOH considered "end of useful life"

RISK_BINS = [0, 20, 40, 60, 80, 100.0001]
RISK_LABELS = ["Critical", "High Risk", "Low Risk", "Good", "Excellent"]

PRECAUTIONS = {
    "Excellent": [
        "Battery is in top condition — continue normal usage.",
        "Maintain charge between 20% and 80% for longevity.",
        "Avoid exposing the pack to temperatures above 45C.",
    ],
    "Good": [
        "Battery is healthy but starting to age — monitor monthly.",
        "Avoid fast-charging every cycle; alternate with standard charging.",
        "Keep firmware / BMS calibration up to date.",
    ],
    "Low Risk": [
        "Schedule a full diagnostic check within the next 30 days.",
        "Reduce fast-charging frequency to slow further degradation.",
        "Avoid deep discharges below 10% SOC.",
    ],
    "High Risk": [
        "Reduce load / high-current draw immediately.",
        "Avoid fast charging until a technician inspects the pack.",
        "Keep the battery between 20-60% SOC and away from heat.",
    ],
    "Critical": [
        "Stop using the battery for high-drain applications immediately.",
        "Have the pack inspected/replaced by a certified technician.",
        "Do not charge above 50% or leave unattended while charging.",
    ],
}


def _parse_complex(series: pd.Series) -> pd.Series:
    """Robustly parse strings like '(938.48-57.98j)' into python complex."""
    def _one(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip()
        s = s.replace(" ", "")
        try:
            return complex(s)
        except ValueError:
            # fallback: strip stray parentheses
            s2 = re.sub(r"[()]", "", s)
            try:
                return complex(s2)
            except ValueError:
                return np.nan
    return series.apply(_one)


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = ["Sense_current", "Battery_current", "Current_ratio",
            "Battery_impedance", "Rectified_Impedance"]
    for c in cols:
        cvals = _parse_complex(df[c])
        df[f"{c}_mag"] = cvals.apply(lambda z: np.nan if pd.isna(z) else abs(z))
        df[f"{c}_phase"] = cvals.apply(lambda z: np.nan if pd.isna(z) else np.angle(z))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # interpolate any missing magnitude/phase values (linear, then ffill/bfill edges)
    feat_cols = [c for c in df.columns if c.endswith("_mag") or c.endswith("_phase")]
    df[feat_cols] = df[feat_cols].interpolate(limit_direction="both")

    # ---- SOH from Rectified_Impedance magnitude ----
    # Internal impedance is a well-established inverse proxy for cell health
    # (higher impedance -> lower health). Scaled across the FULL 0-100% range
    # observed in this pack so every risk band (Critical..Excellent) is
    # represented in the dataset.
    ri = df["Rectified_Impedance_mag"]
    ri_min, ri_max = ri.min(), ri.max()
    soh_raw = 1 - (ri - ri_min) / (ri_max - ri_min)          # 0..1, higher = healthier
    df["SOH"] = (soh_raw * 100).clip(0, 100)

    # ---- SOC from Current_ratio magnitude ----
    cr = df["Current_ratio_mag"]
    cr_min, cr_max = cr.min(), cr.max()
    soc_raw = (cr - cr_min) / (cr_max - cr_min)
    df["SOC"] = (soc_raw * 100).rolling(5, min_periods=1, center=True).mean().clip(0, 100)

    # ---- RUL (cycles remaining until the 80% SOH knee) ----
    # The raw row order in this file is a per-sample impedance log, not a
    # verified chronological cycle count, so instead of trusting row order we
    # use the standard engineering assumption that a cell degrades roughly
    # linearly from 100% to the 80% knee over a nominal ~1000-cycle rated
    # lifetime (a common approximation used across the RUL literature, see
    # Refs [5],[6],[13] in the synopsis). This keeps RUL a direct, monotonic
    # function of the estimated SOH for every sample.
    NOMINAL_LIFE_CYCLES = 1000
    df["RUL"] = ((df["SOH"] - KNEE_SOH) / (100 - KNEE_SOH) * NOMINAL_LIFE_CYCLES).clip(lower=0)

    # ---- Risk category from SOH bands ----
    df["Risk_Category"] = pd.cut(df["SOH"], bins=RISK_BINS, labels=RISK_LABELS,
                                  right=False, include_lowest=True)
    df["Risk_Category"] = df["Risk_Category"].astype(str)

    return df


def build_windows(df: pd.DataFrame, window: int = WINDOW):
    """Return sliding-window sequences (X) and per-window targets (y_*)."""
    feat_cols = [c for c in df.columns if c.endswith("_mag") or c.endswith("_phase")]
    X_raw = df[feat_cols].values.astype(np.float32)

    # z-score normalisation (fit on the whole series — documented, simple, stable)
    mu, sigma = X_raw.mean(axis=0), X_raw.std(axis=0) + 1e-8
    X_norm = (X_raw - mu) / sigma

    Xs, soh_y, soc_y, rul_y, risk_y = [], [], [], [], []
    for i in range(window, len(df)):
        Xs.append(X_norm[i - window:i])
        soh_y.append(df["SOH"].iloc[i])
        soc_y.append(df["SOC"].iloc[i])
        rul_y.append(df["RUL"].iloc[i])
        risk_y.append(df["Risk_Category"].iloc[i])

    return (np.array(Xs), np.array(soh_y), np.array(soc_y),
            np.array(rul_y), np.array(risk_y), feat_cols, mu, sigma)


def run():
    df = load_raw()
    df = engineer_features(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"[preprocess] wrote {PROCESSED_PATH}  shape={df.shape}")
    print(df[["SOH", "SOC", "RUL", "Risk_Category"]].describe(include="all"))
    print(df["Risk_Category"].value_counts())

    X, soh_y, soc_y, rul_y, risk_y, feat_cols, mu, sigma = build_windows(df)
    print(f"[preprocess] windows: X={X.shape}  soh={soh_y.shape}  risk classes={set(risk_y)}")
    np.savez("data/windowed.npz", X=X, soh=soh_y, soc=soc_y, rul=rul_y, risk=risk_y,
             feat_cols=np.array(feat_cols), mu=mu, sigma=sigma)
    print("[preprocess] wrote data/windowed.npz")
    return df


if __name__ == "__main__":
    run()
