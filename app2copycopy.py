# app.py — Cyber Attack Forecasting Tool (Streamlit + XGBoost)
# ——————————————————————————————————————————————————————————————
# - Robust timestamp inference (no more "Attack Start Time missing")
# - Thin-ingest keeps any time-ish column automatically
# - Processes EVERY ROW → builds hourly counts in-memory (lightweight)
# - Trims extreme spikes before training (top-quantile slider)
# - XGBoost per-threat forecaster with recent-mean damping and noise

import os
import re
import glob
import joblib
import warnings
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import shutil
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Optional but handy if you later export snapshots
import pyarrow as pa
import pyarrow.parquet as pq

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.autolayout": True})

# =========================
# ---- CONFIG / STORAGE ----
# =========================
STATELESS_ONLY = True  # session-only roll-up (recommended for big CSVs)
st.set_page_config(page_title="Cyber Attacks Forecaster", page_icon="🛡️", layout="wide")

DATA_DIR      = "data"
MODELS_DIR    = "models"
PLOTS_DIR     = "plots"
PROCESSED_DIR = "processed"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# ===========================
# ---- UTILS / HELPERS  -----
# ===========================

def _normalize_and_uniquify_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    if len(cols) != len(set(cols)):
        seen = {}
        new_cols = []
        for c in cols:
            k = seen.get(c, 0)
            new_cols.append(c if k == 0 else f"{c}.{k}")
            seen[c] = k + 1
        df.columns = new_cols
    else:
        df.columns = cols
    return df

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

TIME_CANDIDATES_EXACT = [
    "Attack Start Time", "Attack Start Time (UTC)",
    "First Seen", "First Seen (UTC)",
    "Start Time", "Event Time", "EventTime",
    "Timestamp", "@timestamp", "Time", "Datetime",
    "Date", "LogTime", "Event Received Time"
]

def _find_time_col(cols) -> str | None:
    # 1) exact/synonym match (case/punct insensitive)
    norm_map = {_norm(c): c for c in cols}
    for name in TIME_CANDIDATES_EXACT:
        k = _norm(name)
        if k in norm_map:
            return norm_map[k]
    # 2) fuzzy tokens
    tokens = ["timestamp", "firstseen", "attackstart", "starttime", "eventtime",
              "time", "datetime", "date", "fecha", "fechahora"]
    for c in cols:
        if any(t in _norm(c) for t in tokens):
            return c
    return None

def _ensure_attack_start_time(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee df['Attack Start Time'] exists and is datetime, from any plausible source."""
    if "Attack Start Time" in df.columns:
        df["Attack Start Time"] = pd.to_datetime(df["Attack Start Time"], errors="coerce")
        return df

    cand = _find_time_col(df.columns)
    if cand is None:
        raise ValueError(
            "Could not find a timestamp column. Include 'Attack Start Time' or a similar field "
            "(e.g., 'First Seen', '@timestamp', 'Event Time', epoch, etc.)."
        )
    s = df[cand]
    # Epoch? Try ms → s → else generic parse
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        dt_ms = pd.to_datetime(s, unit="ms", errors="coerce")
        if dt_ms.notna().any():
            df["Attack Start Time"] = dt_ms
        else:
            df["Attack Start Time"] = pd.to_datetime(s, unit="s", errors="coerce")
    else:
        df["Attack Start Time"] = pd.to_datetime(s, errors="coerce")

    return df

# --------- Thin ingest: keep known + time-ish columns ----------
THIN_INPUT_COLS = {
    "Attack Start Time", "First Seen",
    "Threat Type", "Threat Name", "Threat Subtype",
    "Severity", "Source IP", "Destination IP",
    "Attacker", "Victim", "Addition Info",
    "attack_result", "direction", "duration",
}

def make_usecols_callable(keep_cols: set[str]):
    """Keep known columns AND anything that looks like a timestamp so thin ingest never drops it."""
    lower_keep = {c.lower() for c in keep_cols}
    time_tokens = [
        "time", "timestamp", "@timestamp", "first seen", "first_seen",
        "start time", "event time", "datetime", "fecha", "date", "logtime"
    ]
    def _f(colname: str) -> bool:
        name = str(colname); lname = name.lower()
        if (name in keep_cols) or (lname in lower_keep):
            return True
        if any(tok in lname for tok in time_tokens):
            return True
        return False
    return _f

# ===============================
# ---- 1) INGEST (EVERY ROW) ----
# ===============================

def process_csv_to_hourly_counts(
    input_path: str,
    chunksize: int = 250_000,
    usecols_filter=None,
) -> dict:
    """
    Stream-read the raw CSV in chunks, infer timestamp robustly, and build a compact
    hourly counts DF: ['Threat Type','ds','y'] (ds is hourly).
    Returns dict with:
      - 'counts_path' : CSV path with hourly counts
      - 'rows'        : total raw rows processed
    """
    prog = st.progress(0.0, text="Leyendo CSV…")
    rows_done = 0

    rollup = None  # ["Threat Type","ds","y"]

    reader = pd.read_csv(
        input_path,
        low_memory=False,
        chunksize=chunksize,
        usecols=usecols_filter,  # callable or None
    )

    for i, df in enumerate(reader):
        df = _normalize_and_uniquify_columns(df)
        df = _ensure_attack_start_time(df)

        # Coerce Threat Type to string; missing → "Unknown"
        if "Threat Type" not in df.columns:
            df["Threat Type"] = "Unknown"
        df["Threat Type"] = df["Threat Type"].astype(str).fillna("Unknown")

        # Build hourly bins
        ts = pd.to_datetime(df["Attack Start Time"], errors="coerce")
        g = pd.DataFrame({
            "Threat Type": df["Threat Type"].astype(str),
            "ds": ts.dt.floor("h"),
        })
        g = g.dropna(subset=["ds"])
        g = g.groupby(["Threat Type", "ds"]).size().reset_index(name="y")

        if rollup is None:
            rollup = g
        else:
            rollup = pd.concat([rollup, g], ignore_index=True)
            # periodically collapse to keep small
            if len(rollup) > 300_000:
                rollup = rollup.groupby(["Threat Type", "ds"], as_index=False)["y"].sum()

        rows_done += len(df)
        prog.progress(min(0.99, 0.02 + i * 0.02), text=f"Procesadas ~{rows_done:,} filas")

    if rollup is None:
        rollup = pd.DataFrame(columns=["Threat Type", "ds", "y"])
    else:
        rollup = rollup.groupby(["Threat Type", "ds"], as_index=False)["y"].sum()
    rollup["ds"] = pd.to_datetime(rollup["ds"], errors="coerce")

    # Persist a tiny CSV for download/debug
    counts_out = os.path.join(PROCESSED_DIR, f"hourly_counts_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    rollup.to_csv(counts_out, index=False)

    prog.progress(1.0, text=f"¡Listo! Total procesado: {rows_done:,} filas")
    return {"counts_path": counts_out, "rows": rows_done}

# ==================================
# ---- 2) DATA LAYER (SESSION)  ----
# ==================================

def _append_session_rollup(new_counts: pd.DataFrame) -> pd.DataFrame:
    base = st.session_state.get("session_master_df")
    if base is None or base.empty:
        st.session_state["session_master_df"] = new_counts
    else:
        merged = pd.concat([base, new_counts], ignore_index=True)
        merged = merged.groupby(["Threat Type","ds"], as_index=False)["y"].sum()
        st.session_state["session_master_df"] = merged
    return st.session_state["session_master_df"]

def _session_master_df() -> pd.DataFrame:
    return st.session_state.get("session_master_df", pd.DataFrame())

def build_hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    """If already a roll-up, just normalize; else derive from raw (not used here)."""
    if df.empty:
        return df
    if {"Threat Type","ds","y"}.issubset(df.columns):
        out = df.copy()
        out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
        return out[["Threat Type","ds","y"]]
    # Fallback path (not expected with this app flow)
    tmp = df.copy()
    tmp["ds"] = pd.to_datetime(tmp["Attack Start Time"], errors="coerce").dt.floor("h")
    tmp["Threat Type"] = tmp.get("Threat Type","Unknown").astype(str)
    return tmp.groupby(["Threat Type","ds"]).size().reset_index(name="y")

@st.cache_data(show_spinner=False)
def hourly_counts_cached(df: pd.DataFrame):
    return build_hourly_counts(df)

# ============================================
# ---- 3) FEATURES / MODEL / FORECASTING  ----
# ============================================

WINDOW_CONFIG = {
    "DoS": {"rolling": 3, "lags": [1, 2]},
    "Scan": {"rolling": 6, "lags": [1, 2, 6]},
    "Malicious Flow": {"rolling": 12, "lags": [1, 2, 6]},
    "Vulnerability Attack": {"rolling": 6, "lags": [1, 2, 24]},
    "Attack": {"rolling": 6, "lags": [1, 2]},
    "Malfile": {"rolling": 3, "lags": [1, 2]},
}

def _merge_extra_columns(grouped: pd.DataFrame, raw_df: pd.DataFrame, threat: str) -> pd.DataFrame:
    """We only have roll-up; add neutral columns so the model shape is stable."""
    merged = grouped.copy()
    for c in ["Severity", "attack_result_label", "direction", "duration"]:
        merged[c] = 0
    return merged

def _add_time_features(subset: pd.DataFrame) -> pd.DataFrame:
    subset = subset.copy()
    subset["y_log"] = np.log1p(subset["y"])
    subset["hour"] = subset["ds"].dt.hour
    subset["dayofweek"] = subset["ds"].dt.dayofweek
    subset["is_weekend"] = subset["dayofweek"].isin([5, 6]).astype(int)
    subset["is_night"] = subset["hour"].apply(lambda x: 1 if x < 7 or x > 21 else 0)
    subset["weekofyear"] = subset["ds"].dt.isocalendar().week.astype(int)
    subset["time_since_last"] = subset["ds"].diff().dt.total_seconds().div(3600).fillna(0)
    return subset

def _add_lags_rolls(subset: pd.DataFrame, threat: str):
    subset = subset.copy()
    base_cfg = {"rolling": 3, "lags": [1, 2, 6]}
    cfg_src = WINDOW_CONFIG.get(threat, base_cfg)
    cfg = {
        "rolling": int(cfg_src.get("rolling", base_cfg["rolling"])),
        "lags": [int(l) for l in cfg_src.get("lags", base_cfg["lags"]) if int(l) in (1, 2, 6)],
    }
    if not cfg["lags"]:
        cfg["lags"] = [1, 2, 6]

    for lag in cfg["lags"]:
        subset[f"lag{lag}"] = subset["y_log"].shift(lag)

    subset["rolling_mean"] = subset["y_log"].rolling(cfg["rolling"]).mean().shift(1)
    subset["rolling_std"]  = subset["y_log"].rolling(cfg["rolling"]).std().shift(1)

    # Ensure no stray column remains from past versions
    if "rolling_sum24h" in subset.columns:
        subset = subset.drop(columns=["rolling_sum24h"])

    return subset, cfg

def _enough_history(subset, horizon_hours: int) -> bool:
    return len(subset) >= max(200, int(horizon_hours * 3))

def train_xgb_for_threat(master_df: pd.DataFrame, threat: str, clip_q: float = 0.98, test_days: int = 7):
    grouped = build_hourly_counts(master_df)
    if grouped.empty:
        return None

    sub = grouped[grouped["Threat Type"] == threat].copy()
    if len(sub) < 150:
        return None

    # Trim extreme spikes BEFORE feature building (you asked for top 2%)
    thr = sub["y"].quantile(float(clip_q))
    sub = sub[sub["y"] <= thr]

    sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
    sub = _add_time_features(sub)
    sub = _merge_extra_columns(sub, master_df, threat)
    sub, cfg = _add_lags_rolls(sub, threat)

    feature_cols = [
        "hour","dayofweek","is_weekend","is_night","weekofyear","time_since_last",
        "Severity","attack_result_label","direction","duration",
    ]
    feature_cols += [c for c in ["lag1","lag2","lag6","lag24"] if c in sub.columns]
    for c in ["rolling_mean", "rolling_std"]:
        if c in sub.columns:
            feature_cols.append(c)

    sub = sub.dropna(subset=feature_cols)
    if sub.empty:
        return None

    cutoff = sub["ds"].max() - pd.Timedelta(days=int(test_days))
    train = sub[sub["ds"] <= cutoff]
    test  = sub[sub["ds"] >  cutoff]
    if len(train) < 50 or len(test) < 20:
        return None

    X_full = train[feature_cols]
    y_full = train["y_log"]
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        objective="reg:squarederror", n_jobs=4,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred_log = model.predict(X_val)
    val_mae  = float(np.mean(np.abs(np.expm1(val_pred_log) - np.expm1(y_val))))
    val_rmse = float(np.sqrt(np.mean((np.expm1(val_pred_log) - np.expm1(y_val))**2)))
    resid_std_log = float(np.std(y_val - val_pred_log))

    model_path = os.path.join(MODELS_DIR, f"xgb_{re.sub('[^A-Za-z0-9]+','_', threat)}.joblib")
    joblib.dump(
        {"model": model, "features": feature_cols, "cfg": cfg, "resid_std_log": resid_std_log},
        model_path,
    )

    return {
        "model_path": model_path,
        "features": feature_cols,
        "cfg": cfg,
        "train": train,
        "test": test,
        "validation": {"val_mae": val_mae, "val_rmse": val_rmse, "resid_std_log": resid_std_log},
    }

def forecast_recursive(
    master_df: pd.DataFrame,
    threat: str,
    horizon_days: int,
    model_bundle: dict,
    seasonality_strength: float | None = None,
    noise_level: float | None = None,
    spike_prob: float | None = None,
    seed: int = 1234,
):
    rng = np.random.default_rng(seed)

    grouped = build_hourly_counts(master_df)
    sub = grouped[grouped["Threat Type"] == threat].copy()
    sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
    sub = _add_time_features(sub)
    sub = _merge_extra_columns(sub, master_df, threat)
    sub, _ = _add_lags_rolls(sub, threat)

    feature_cols  = model_bundle["features"]
    model         = model_bundle["model"]
    resid_std_log = float(model_bundle.get("resid_std_log", 0.10))

    history = sub.dropna(subset=feature_cols).copy().sort_values("ds")
    if history.empty:
        return None

    horizon_hours = int(horizon_days * 24)
    if not _enough_history(history, horizon_hours):
        return {"insufficient_history": True, "needed": max(200, horizon_hours * 3), "available": len(history)}

    # Auto-tune if not provided
    if seasonality_strength is None or noise_level is None or spike_prob is None:
        tail = history.tail(min(len(history), 24*14))
        by_hour = tail.groupby(tail["ds"].dt.hour)["y_log"].mean()
        amp = float(by_hour.max() - by_hour.min()) if not by_hour.empty else 0.0
        seasonality_strength = float(np.clip(amp / 1.2, 0.30, 0.85)) if seasonality_strength is None else seasonality_strength
        noise_level = float(np.clip(resid_std_log / 0.35, 0.15, 0.60)) if noise_level is None else noise_level

        y_tail = np.expm1(tail["y_log"])
        if len(y_tail) >= 24:
            med = float(np.median(y_tail))
            mad = float(np.median(np.abs(y_tail - med))) + 1e-9
            spikes = int(np.sum(y_tail > med + 6 * mad))
            sp_est = spikes / max(1, len(y_tail))
        else:
            sp_est = 0.01
        spike_prob = float(np.clip(sp_est, 0.005, 0.08)) if spike_prob is None else spike_prob

    max_lag = max([int(x[3:]) for x in feature_cols if x.startswith("lag")] + [1])
    buffer_points = max(max_lag + 24, 30)
    hist_tail = history.tail(buffer_points).copy()
    ylog_series = hist_tail["y_log"].tolist()
    ds_last = history["ds"].max()

    rows = []
    for _ in range(horizon_hours):
        ds_next = ds_last + pd.Timedelta(hours=1)
        hour = ds_next.hour
        dayofweek = ds_next.dayofweek
        is_weekend = int(dayofweek in [5, 6])
        is_night = 1 if hour < 7 or hour > 21 else 0
        weekofyear = int(pd.Timestamp(ds_next).isocalendar().week)

        row = {
            "ds": ds_next,
            "hour": hour, "dayofweek": dayofweek,
            "is_weekend": is_weekend, "is_night": is_night,
            "weekofyear": weekofyear,
            "time_since_last": 1.0,
            "Severity": 0, "attack_result_label": 0, "direction": 0, "duration": 0,
        }
        for c in feature_cols:
            if c.startswith("lag"):
                k = int(c[3:])
                row[c] = ylog_series[-k] if len(ylog_series) >= k else np.nan

        roll_n = model_bundle["cfg"].get("rolling", 3)
        if len(ylog_series) >= roll_n:
            row["rolling_mean"] = float(pd.Series(ylog_series[-roll_n:]).mean())
            row["rolling_std"]  = float(pd.Series(ylog_series[-roll_n:]).std(ddof=0))
        else:
            row["rolling_mean"] = np.nan
            row["rolling_std"]  = np.nan

        X_row = pd.DataFrame([row]).dropna(subset=feature_cols)
        base_pred_log = ylog_series[-1] if X_row.empty else float(model.predict(X_row[feature_cols])[0])

        recent_mean_log = float(pd.Series(ylog_series[-24:]).mean()) if len(ylog_series) >= 24 else float(np.mean(ylog_series))
        blended_log = seasonality_strength * base_pred_log + (1.0 - seasonality_strength) * recent_mean_log

        noise = rng.normal(loc=0.0, scale=resid_std_log * noise_level)
        if rng.random() < spike_prob:
            noise += rng.normal(0.0, resid_std_log * 2.5 * noise_level)

        y_pred_log = blended_log + noise
        ylog_series.append(y_pred_log)
        rows.append({"ds": ds_next, "y_hat": float(np.expm1(y_pred_log))})
        ds_last = ds_next

    return pd.DataFrame(rows)

def plot_recent_and_forecast(threat: str, recent_actual: pd.DataFrame, fcst_df: pd.DataFrame, lookback_hours: int = 48, title_suffix: str = ""):
    fig, ax = plt.subplots(figsize=(12, 5))
    if recent_actual is not None and not recent_actual.empty:
        ax.plot(recent_actual["ds"], recent_actual["y"], label=f"Actual (last {lookback_hours}h)")
    if fcst_df is not None and not fcst_df.empty:
        ax.plot(fcst_df["ds"], fcst_df["y_hat"], label="Forecast", linewidth=2)
    ax.set_title(f"{threat} — Hourly Attacks {title_suffix}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Count")
    ax.legend()
    return fig

@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    return joblib.load(path)

# ===========================
# ---- 4) STREAMLIT UI  -----
# ===========================

st.title("🛡️ Predicción de Ataques")
st.caption("Subir Información → Procesar (cada fila) → Entrenar → Predecir")

# Sidebar: current data status + controls
with st.sidebar:
    st.header("📦 Data Status")
    master = _session_master_df()
    if not master.empty and {"Threat Type","ds","y"}.issubset(master.columns):
        start = pd.to_datetime(master["ds"], errors="coerce").min()
        end   = pd.to_datetime(master["ds"], errors="coerce").max()
        st.success(f"Data from **{start.date()}** to **{end.date()}**  \nRows (hour bins): **{len(master):,}**")
        tt_list = sorted(map(str, master["Threat Type"].dropna().unique()))
        st.write(f"Threat Types ({len(tt_list)}):")
        st.write(", ".join(tt_list[:30]) + (" ..." if len(tt_list) > 30 else ""))
        # download current session roll-up
        snap = os.path.join(DATA_DIR, "session_master.csv")
        master.to_csv(snap, index=False)
        st.download_button("⬇️ Download session master.csv", data=open(snap,"rb").read(),
                           file_name="session_master.csv", mime="text/csv")
    else:
        st.info("No data yet. Upload or fetch to get started.")

    st.markdown("---")
    st.caption("Hourly features")
    st.button("Build from raw rows (uses every row)", disabled=True)
    clip_q = st.slider(
        "Recorte superior (cuantil)",
        min_value=0.90, max_value=0.999, value=0.98, step=0.001,
        help="Recorta picos antes de entrenar (por amenaza)."
    )
    st.session_state["clip_q"] = float(clip_q)

    st.markdown("---")
    if st.button("↻ Clear caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.session_state.pop("session_master_df", None)
        st.success("Caches cleared. Reloading…")
        st.experimental_rerun()

# -----------------------
# 1) Upload & Processing
# -----------------------
st.subheader("1) Agrega Información para Entrenar el Modelo")
st.write("Sube un CSV sin procesar → se **procesará** y se **agregará** al **dataset de sesión** (roll-up por hora).")

uploaded = st.file_uploader("Subir CSV (exportación BDS u otra)", type=["csv"])

thin_ingest = st.toggle(
    "🪶 Thin ingest (usar solo columnas relevantes + cualquier columna de tiempo)",
    value=True,
    help="Mantiene columnas clave y cualquier columna que parezca timestamp; acelera lecturas grandes."
)
chunksize_opt = st.select_slider(
    "Tamaño de chunk para procesar",
    options=[100_000, 150_000, 200_000, 250_000, 300_000],
    value=250_000,
    format_func=lambda x: f"{x:,} filas",
)

# Optional URL ingest (Dropbox/Drive/HTTP)
st.markdown("**O pega un enlace directo (Dropbox / Google Drive / HTTPS):**")
url_in = st.text_input("URL a un CSV (o .gz/.zip con un CSV dentro)", placeholder="https://…")
fetch_btn = st.button("Fetch & Merge from URL", use_container_width=True, disabled=not url_in)

def _handle_ingest(csv_path: str):
    before = _session_master_df()
    before_bins = len(before) if not before.empty else 0

    with st.status("Reading & enriching (session mode)…", expanded=True) as status:
        usecols_cb = make_usecols_callable(THIN_INPUT_COLS) if thin_ingest else None
        result = process_csv_to_hourly_counts(csv_path, chunksize=chunksize_opt, usecols_filter=usecols_cb)

        counts_df = pd.read_csv(result["counts_path"])
        counts_df["ds"] = pd.to_datetime(counts_df["ds"], errors="coerce")

        merged = _append_session_rollup(counts_df)
        after_bins = len(merged)

        status.update(label="Merge complete ✅ (session only; nothing persisted permanently)", state="complete")

    st.metric("Events ingested (raw rows)", value=f"{result['rows']:,}")
    st.metric("Hourly bins in memory", value=f"{after_bins:,}", delta=f"+{after_bins - before_bins:,}")

if fetch_btn and url_in:
    # Minimal downloader (no special Google Drive handling here to keep code short)
    import requests, tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmpf:
        with st.spinner("Downloading…"):
            r = requests.get(url_in, timeout=600)
            r.raise_for_status()
            tmpf.write(r.content)
        tmp_path = tmpf.name
    _handle_ingest(tmp_path)

colA, colB = st.columns([1,1])
with colA:
    st.write("")  # spacer
with colB:
    process_btn = st.button("Process & Merge (from uploaded file)", type="primary", use_container_width=True, disabled=uploaded is None)

if process_btn and uploaded is not None:
    raw_path = os.path.join(DATA_DIR, f"upload_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    uploaded.seek(0)
    with open(raw_path, "wb") as dst:
        shutil.copyfileobj(uploaded, dst, length=16 * 1024 * 1024)
    _handle_ingest(raw_path)

st.divider()

# -----------------------
# 2) Training & Forecast
# -----------------------
st.subheader("2) Entrenar el modelo y generar predicciones")

master = _session_master_df()
if master.empty:
    st.warning("Cargue y procese al menos un CSV primero.")
else:
    threats = sorted(map(str, master["Threat Type"].dropna().unique()))
    chosen = st.multiselect("Elija los tipos de amenazas para entrenar", options=threats, default=threats[:1])

    horizon_choice = st.select_slider("Horizonte de previsión", options=[7, 14, 30], value=7, format_func=lambda d: f"{d} days")
    lookback_hours = st.select_slider(
        "Historial real para mostrar antes del pronóstico",
        options=[24, 48, 72, 96, 120, 144, 168],
        value=48,
        format_func=lambda h: f"{h//24} días",
    )

    run_btn = st.button("Entrene y Pronostico", type="primary", use_container_width=True, disabled=len(chosen) == 0)

    if run_btn:
        grouped_all = hourly_counts_cached(master)
        for threat in chosen:
            with st.spinner(f"Entrenando {threat}…"):
                bundle = train_xgb_for_threat(master, threat, clip_q=st.session_state.get("clip_q", 0.98), test_days=7)
            if bundle is None:
                st.error(f"No hay suficientes datos válidos para **{threat}**.")
                continue

            saved = load_model_cached(bundle["model_path"])
            model_bundle = {
                "model": saved["model"],
                "features": saved["features"],
                "cfg": saved["cfg"],
                "resid_std_log": saved.get("resid_std_log", 0.10),
            }

            fcst = forecast_recursive(master, threat, horizon_days=int(horizon_choice), model_bundle=model_bundle)
            if isinstance(fcst, dict) and fcst.get("insufficient_history"):
                st.warning(
                    f"**{threat}**: No hay suficiente historial con funciones listas para usar {horizon_choice}d "
                    f"(needed ~{fcst['needed']}, available {fcst['available']}). Prueba con un horizonte menor."
                )
                continue

            grouped = grouped_all[grouped_all["Threat Type"] == threat].copy()
            grouped["ds"] = pd.to_datetime(grouped["ds"], errors="coerce")

            fcst_start = fcst["ds"].min() if fcst is not None and not fcst.empty else grouped["ds"].max() + pd.Timedelta(hours=1)
            recent_actual = grouped[
                (grouped["ds"] >= fcst_start - pd.Timedelta(hours=lookback_hours)) & (grouped["ds"] < fcst_start)
            ]

            fig = plot_recent_and_forecast(
                threat,
                recent_actual,
                fcst,
                lookback_hours=lookback_hours,
                title_suffix=f"(+{horizon_choice}d)",
            )
            st.pyplot(fig)

            # Save artifacts for download
            plot_path = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_choice}d.png")
            fig.savefig(plot_path, dpi=160)
            csv_path = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_choice}d_fcst.csv")
            if fcst is not None and not fcst.empty:
                fcst.to_csv(csv_path, index=False)

            val = bundle["validation"]
            st.caption(f"Validation (internal, last split): MAE={val['val_mae']:.2f}  |  RMSE={val['val_rmse']:.2f}")

            c1, c2 = st.columns(2)
            with c1:
                if os.path.exists(csv_path):
                    st.download_button(
                        "⬇️ Download forecast CSV",
                        data=open(csv_path, "rb").read(),
                        file_name=os.path.basename(csv_path),
                        mime="text/csv",
                    )
            with c2:
                st.download_button(
                    "⬇️ Download plot PNG",
                    data=open(plot_path, "rb").read(),
                    file_name=os.path.basename(plot_path),
                    mime="image/png",
                )

st.divider()
st.subheader("Notas & Guardrails")
st.markdown(
    """
- **Se procesa cada fila** del CSV y se agrega por hora (Threat Type × hora).
- **Recorte de atípicos**: slider para recortar el **cuantil superior** (por defecto 0.98) **antes de entrenar**.
- **Horizontes**: 7/14/30 días con verificación de historial suficiente.
- **Modo sesión**: los datos viven en memoria; descarga `session_master.csv` si quieres persistirlos.
"""
)
st.caption("© Streamlit + XGBoost")
