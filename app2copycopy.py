# app.py — Cyber Attack Forecasting Tool (Streamlit + XGBoost)
# Full build: streaming ingest (every row) → hourly counts in session →
# training (with clipping) → recursive forecast (damped + noise + spikes).
# Robust timestamp detection & optional persistent parquet-part machinery.

import os, re, glob, shutil, warnings, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.autolayout": True})

# =========================
# ---- CONFIG / STORAGE ----
# =========================
st.set_page_config(page_title="Cyber Attacks Forecaster", page_icon="🛡️", layout="wide")

# Toggle to keep *only* session memory (no persistent master parts written)
STATELESS_ONLY = True

DATA_DIR       = "data"
MODELS_DIR     = "models"
PLOTS_DIR      = "plots"
PROCESSED_DIR  = "processed"
SEEDS_DIR      = "seeds"
MASTER_DS_DIR  = os.path.join(DATA_DIR, "master_parquet")   # persistent parquet parts
SESSION_SNAP   = os.path.join(DATA_DIR, "session_master.parquet")
SESSION_CSV    = os.path.join(DATA_DIR, "session_master.csv")
ENRICH_SUFFIX_PARQUET = "_enriched_raw.parquet"

for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR, PROCESSED_DIR, SEEDS_DIR, MASTER_DS_DIR]:
    os.makedirs(d, exist_ok=True)

# Minimal columns when thin-ingesting CSVs (case-insensitive matching)
THIN_INPUT_COLS = {
    "Attack Start Time", "First Seen", "Start Time", "Event Time", "Timestamp", "Time",
    "Threat Type", "Threat Name", "Threat Subtype", "Severity",
    "Source IP", "Destination IP", "Attacker", "Victim",
    "Addition Info", "attack_result", "direction", "duration",
}

# =========================
# ---- UTILITIES ----------
# =========================
def make_usecols_callable(keep_cols: set[str]):
    lower_keep = {c.lower() for c in keep_cols}
    def _f(colname: str) -> bool:
        return (colname in keep_cols) or (str(colname).lower() in lower_keep)
    return _f

def _normalize_and_uniquify_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    if len(cols) != len(set(cols)):
        seen, new_cols = {}, []
        for c in cols:
            k = seen.get(c, 0)
            new_cols.append(c if k == 0 else f"{c}.{k}")
            seen[c] = k + 1
        df.columns = new_cols
    else:
        df.columns = cols
    return df

def coalesce_columns(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    for c in extra.columns:
        if c in base.columns:
            base[c] = base[c].where(base[c].notna(), extra[c])
        else:
            base[c] = extra[c]
    return base

def get_first_series(df: pd.DataFrame, name: str):
    if name not in df.columns:
        return None
    obj = df[name]
    return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj

# =========================
# ---- TIME HANDLING  -----
# =========================
_TIME_CANDIDATES = [
    "Attack Start Time", "First Seen", "Start Time", "Event Time", "EventTime",
    "timestamp", "time", "datetime", "date", "log_time", "occurred_at",
    "Time Generated", "Receive Time", "generated_time", "StartTime",
]

def _ensure_attack_start_time(df: pd.DataFrame) -> pd.DataFrame:
    """Create/normalize 'Attack Start Time' from many possible columns (case-insensitive)."""
    df = df.copy()
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    candidate_actual = None
    for cand in _TIME_CANDIDATES:
        if cand.lower() in lower_map:
            candidate_actual = lower_map[cand.lower()]
            break

    if candidate_actual is None:
        # Heuristic: any column whose name contains 'time' or 'date'
        for c in df.columns:
            cl = str(c).lower()
            if "time" in cl or "date" in cl or "stamp" in cl:
                candidate_actual = c
                break

    if candidate_actual is None:
        raise ValueError(
            "No timestamp column found. Expected one of: "
            + ", ".join(_TIME_CANDIDATES)
            + ". Available: "
            + ", ".join(map(str, df.columns[:40]))
        )

    ts = pd.to_datetime(df[candidate_actual], errors="coerce", utc=False, infer_datetime_format=True)
    if ts.isna().mean() > 0.5:
        # Try alternate parsing styles if the first pass failed a lot
        ts = pd.to_datetime(df[candidate_actual], errors="coerce", utc=False, dayfirst=True)
    df["Attack Start Time"] = ts
    return df

# =========================
# ---- ENRICHMENT ----------
# =========================
_addinfo_re = re.compile(r'type=(?P<key>[^ \t]+)\s+value=(?P<val>[^;,\n]+)')

def parse_addition_info_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Addition Info" not in df.columns:
        df["Addition Info"] = np.nan
        return df
    s = df["Addition Info"].fillna("")
    ext = (
        s.str.extractall(_addinfo_re)
        .reset_index()
        .rename(columns={"level_0": "row", "key": "k", "val": "v"})
    )
    if ext.empty:
        return df
    wide = ext.pivot(index="row", columns="k", values="v")
    wide.columns = [str(c).strip() for c in wide.columns]
    wide = wide.reset_index()
    out = (
        df.reset_index(drop=True)
          .reset_index()
          .merge(wide, left_on="index", right_on="row", how="left")
          .drop(columns=["index", "row"])
    )
    return out

def map_attack_result(df: pd.DataFrame) -> pd.DataFrame:
    result_map = {"1": "Attempted", "2": "Successful", 1: "Attempted", 2: "Successful"}
    s = get_first_series(df, "attack_result")
    df["attack_result_label"] = s.map(result_map) if s is not None else np.nan
    return df

def create_attack_signature(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["Threat Name", "Threat Type", "Threat Subtype", "Severity",
            "Source IP", "Destination IP", "Attacker", "Victim"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    df["attack_signature"] = df[cols].astype(str).agg("|".join, axis=1)
    return df

# =========================
# ---- SESSION MASTER -----
# =========================
def _session_counts() -> pd.DataFrame:
    return st.session_state.get("session_counts_df", pd.DataFrame(columns=["Threat Type","ds","y"]))

def _set_session_counts(df: pd.DataFrame):
    st.session_state["session_counts_df"] = df

def _session_downloads():
    df = _session_counts().copy()
    if df.empty:
        return None, None
    try:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), SESSION_SNAP, compression="zstd")
    except Exception:
        pass
    try:
        df.to_csv(SESSION_CSV, index=False)
    except Exception:
        pass
    return (SESSION_SNAP if os.path.exists(SESSION_SNAP) else None,
            SESSION_CSV if os.path.exists(SESSION_CSV) else None)

def _coverage_stats_counts(df: pd.DataFrame):
    if df.empty: return None
    ts = pd.to_datetime(df["ds"], errors="coerce").dropna()
    if ts.empty: return None
    return ts.min().date(), ts.max().date(), len(df)

# =========================
# ---- PERSISTENT MASTER (OPTIONAL) ----
# =========================
def _add_part_to_master(parquet_path: str):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(MASTER_DS_DIR, f"part-{ts}.parquet")
    shutil.copyfile(parquet_path, dst)
    return dst

def _read_master_parquet_unified() -> pd.DataFrame:
    """Reads all parquet parts in MASTER_DS_DIR with schema unification."""
    parts = sorted(glob.glob(os.path.join(MASTER_DS_DIR, "*.parquet")))
    if not parts:
        return pd.DataFrame()
    # Scan schemas
    def norm_type(t: pa.DataType) -> pa.DataType:
        if pa.types.is_dictionary(t): return t.value_type
        return t
    def tag(t: pa.DataType) -> str:
        t = norm_type(t)
        if pa.types.is_string(t) or pa.types.is_large_string(t): return "str"
        if pa.types.is_timestamp(t): return "ts"
        if pa.types.is_floating(t): return "float"
        if pa.types.is_integer(t): return "int"
        if pa.types.is_boolean(t): return "bool"
        return "other"

    observed, samples = {}, {}
    good = []
    for p in parts:
        try:
            t = pq.read_table(p)
            for f in t.schema:
                observed.setdefault(f.name, set()).add(tag(f.type))
                samples.setdefault(f.name, norm_type(f.type))
            good.append(p)
        except Exception:
            continue
    if not good:
        return pd.DataFrame()

    targets: dict[str, pa.DataType] = {}
    for name, tags in observed.items():
        if name == "Attack Start Time":
            targets[name] = pa.large_string()
            continue
        if tags == {"int"}:
            targets[name] = pa.int64()
        elif tags == {"float"} or tags == {"int", "float"}:
            targets[name] = pa.float64()
        elif tags == {"ts"}:
            targets[name] = pa.timestamp("ns")
        elif "str" in tags:
            targets[name] = pa.large_string()
        else:
            targets[name] = pa.large_string()

    tables = []
    for p in good:
        t = pq.read_table(p)
        arrays, names = [], []
        n = t.num_rows
        for name, target in targets.items():
            if name in t.column_names:
                col = t[name]
                if hasattr(col, "chunks"):
                    chs = [pc.cast(ch, target) for ch in col.chunks]
                    arr = pa.chunked_array(chs, type=target)
                else:
                    arr = pc.cast(col, target)
            else:
                arr = pa.nulls(n, type=target)
            arrays.append(arr); names.append(name)
        tables.append(pa.table(arrays, names=names))

    if not tables:
        return pd.DataFrame()

    table = pa.concat_tables(tables, promote=True)
    df = table.to_pandas()
    if "Attack Start Time" in df.columns:
        df["Attack Start Time"] = pd.to_datetime(df["Attack Start Time"], errors="coerce")
    return df

# =========================
# ---- STREAMING INGEST ----
# =========================
def process_csv_to_hourly_counts(csv_path: str, chunksize: int = 250_000, usecols_filter=None):
    """
    Stream-read the CSV in chunks and aggregate to hourly counts.
    Returns (hourly_df, rows_done). *Every row* is used.
    """
    prog = st.progress(0.0, text="Leyendo CSV…")
    rows_done = 0
    hourly = None

    for i, df in enumerate(pd.read_csv(csv_path, low_memory=False, chunksize=chunksize, usecols=usecols_filter)):
        df = _normalize_and_uniquify_columns(df)
        df = _ensure_attack_start_time(df)     # ← FIX: tolerant timestamp detection
        df = parse_addition_info_column(df)
        df = map_attack_result(df)
        df = create_attack_signature(df)

        ts = pd.to_datetime(df["Attack Start Time"], errors="coerce")
        df["ds"] = ts.dt.floor("h")

        tt = df.get("Threat Type")
        if tt is None:
            # try another typical variant
            tt = df.get("threat_type") if "threat_type" in df.columns else pd.Series([""] * len(df))
        df["Threat Type"] = tt.astype(str)

        g = (
            df.dropna(subset=["ds"])
              .groupby(["Threat Type","ds"])
              .size()
              .reset_index(name="y")
        )

        if hourly is None:
            hourly = g
        else:
            hourly = pd.concat([hourly, g], ignore_index=True)
            if len(hourly) > 300_000:
                hourly = hourly.groupby(["Threat Type","ds"], as_index=False)["y"].sum()

        rows_done += len(df)
        prog.progress(min(0.99, 0.02 + i * 0.02), text=f"Procesadas ~{rows_done:,} filas")

    if hourly is None:
        hourly = pd.DataFrame(columns=["Threat Type","ds","y"])
    else:
        hourly = hourly.groupby(["Threat Type","ds"], as_index=False)["y"].sum()
        hourly["ds"] = pd.to_datetime(hourly["ds"], errors="coerce")

    prog.progress(1.0, text=f"¡Listo! Total procesado: {rows_done:,} filas")
    return hourly, rows_done

# =========================
# ---- FEATURES / MODEL ----
# =========================
WINDOW_CONFIG = {
    "DoS": {"rolling": 3, "lags": [1, 2]},
    "Scan": {"rolling": 6, "lags": [1, 2, 6]},
    "Malicious Flow": {"rolling": 12, "lags": [1, 2, 6]},
    "Vulnerability Attack": {"rolling": 6, "lags": [1, 2, 24]},
    "Attack": {"rolling": 6, "lags": [1, 2]},
    "Malfile": {"rolling": 3, "lags": [1, 2]},
}

def _add_time_features(sub: pd.DataFrame) -> pd.DataFrame:
    sub = sub.copy()
    sub["y"] = pd.to_numeric(sub["y"], errors="coerce").fillna(0)
    sub["y_log"] = np.log1p(sub["y"])
    sub["hour"] = sub["ds"].dt.hour
    sub["dayofweek"] = sub["ds"].dt.dayofweek
    sub["is_weekend"] = sub["dayofweek"].isin([5,6]).astype(int)
    sub["is_night"] = (sub["hour"].lt(7) | sub["hour"].gt(21)).astype(int)
    sub["weekofyear"] = sub["ds"].dt.isocalendar().week.astype(int)
    sub["time_since_last"] = sub["ds"].diff().dt.total_seconds().div(3600).fillna(0)
    return sub

def _add_lags_rolls(sub: pd.DataFrame, threat: str):
    sub = sub.copy()
    cfg_src = WINDOW_CONFIG.get(threat, {"rolling": 3, "lags": [1,2,6]})
    cfg = {"rolling": int(cfg_src.get("rolling", 3)),
           "lags": [int(l) for l in cfg_src.get("lags", [1,2,6]) if int(l) in (1,2,6,24)]}
    if not cfg["lags"]:
        cfg["lags"] = [1,2,6]
    for lag in cfg["lags"]:
        sub[f"lag{lag}"] = sub["y_log"].shift(lag)
    sub["rolling_mean"] = sub["y_log"].rolling(cfg["rolling"]).mean().shift(1)
    sub["rolling_std"]  = sub["y_log"].rolling(cfg["rolling"]).std().shift(1)
    return sub, cfg

def _enough_history(sub, horizon_hours: int) -> bool:
    return len(sub) >= max(200, int(horizon_hours * 3))

def train_xgb_for_threat(hourly_all: pd.DataFrame, threat: str, clip_q: float = 0.98, test_days: int = 7):
    sub = hourly_all[hourly_all["Threat Type"] == threat].copy()
    if len(sub) < 150:
        return None

    # Clip extreme spikes (you control clip_q in the UI)
    if 0.5 < clip_q < 1.0 and "y" in sub.columns and len(sub) > 10:
        thr = sub["y"].quantile(clip_q)
        sub = sub[sub["y"] <= thr]

    sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
    sub = _add_time_features(sub)
    sub, cfg = _add_lags_rolls(sub, threat)

    feature_cols = [
        "hour","dayofweek","is_weekend","is_night","weekofyear","time_since_last",
    ] + [c for c in ["lag1","lag2","lag6","lag24","rolling_mean","rolling_std"] if c in sub.columns]

    sub = sub.dropna(subset=feature_cols)
    if sub.empty:
        return None

    cutoff = sub["ds"].max() - pd.Timedelta(days=test_days)
    train = sub[sub["ds"] <= cutoff]
    test  = sub[sub["ds"] >  cutoff]
    if len(train) < 50 or len(test) < 20:
        return None

    X_full = train[feature_cols]
    y_full = train["y_log"]
    X_tr, X_val, y_tr, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        objective="reg:squarederror", n_jobs=4,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    val_pred_log = model.predict(X_val)
    val_mae = float(np.mean(np.abs(np.expm1(val_pred_log) - np.expm1(y_val))))
    val_rmse = float(np.sqrt(np.mean((np.expm1(val_pred_log) - np.expm1(y_val))**2)))
    resid_std_log = float(np.std(y_val - val_pred_log))

    model_path = os.path.join(MODELS_DIR, f"xgb_{re.sub('[^A-Za-z0-9]+','_', threat)}.joblib")
    joblib.dump({"model": model, "features": feature_cols, "cfg": cfg, "resid_std_log": resid_std_log}, model_path)

    return {
        "model_path": model_path,
        "features": feature_cols,
        "cfg": cfg,
        "train": train,
        "test": test,
        "validation": {"val_mae": val_mae, "val_rmse": val_rmse, "resid_std_log": resid_std_log},
    }

def forecast_recursive(hourly_all: pd.DataFrame, threat: str, horizon_days: int, model_bundle: dict,
                       seasonality_strength: float = 0.60, noise_level: float = 0.35, spike_prob: float = 0.02, seed: int = 1234):
    rng = np.random.default_rng(seed)

    sub = hourly_all[hourly_all["Threat Type"] == threat].copy()
    sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
    sub = _add_time_features(sub)
    sub, cfg = _add_lags_rolls(sub, threat)

    feature_cols = model_bundle["features"]
    model = model_bundle["model"]
    resid_std_log = float(model_bundle.get("resid_std_log", 0.10))

    history = sub.dropna(subset=feature_cols).copy().sort_values("ds")
    if history.empty:
        return None

    # Simple auto-tune if defaults are None
    tail = history.tail(min(len(history), 24*14))
    by_hour = tail.groupby(tail["ds"].dt.hour)["y_log"].mean()
    if not by_hour.empty and seasonality_strength is None:
        amp = float(by_hour.max() - by_hour.min())
        seasonality_strength = float(np.clip(amp / 1.2, 0.30, 0.85))
    if noise_level is None:
        noise_level = float(np.clip(resid_std_log / 0.35, 0.15, 0.60))
    if spike_prob is None:
        y_tail = np.expm1(tail["y_log"])
        if len(y_tail) >= 24:
            med = float(np.median(y_tail)); mad = float(np.median(np.abs(y_tail - med))) + 1e-9
            sp_est = np.mean(y_tail > med + 6 * mad)
        else:
            sp_est = 0.01
        spike_prob = float(np.clip(sp_est, 0.005, 0.08))

    horizon_hours = int(horizon_days * 24)
    if not _enough_history(history, horizon_hours):
        return {"insufficient_history": True, "needed": max(200, horizon_hours*3), "available": len(history)}

    max_lag = max([int(x[3:]) for x in feature_cols if x.startswith("lag")] + [1])
    buffer_points = max(max_lag + 24, 30)
    hist_tail = history.tail(buffer_points).copy()
    ylog_series = hist_tail["y_log"].tolist()
    ds_last = history["ds"].max()

    rows = []
    for _ in range(horizon_hours):
        ds_next = ds_last + pd.Timedelta(hours=1)
        row = {
            "ds": ds_next,
            "hour": ds_next.hour,
            "dayofweek": ds_next.dayofweek,
            "is_weekend": int(ds_next.dayofweek in [5,6]),
            "is_night": int(ds_next.hour < 7 or ds_next.hour > 21),
            "weekofyear": int(pd.Timestamp(ds_next).isocalendar().week),
            "time_since_last": 1.0,
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
        damp = seasonality_strength if seasonality_strength is not None else 0.60
        blended_log = damp * base_pred_log + (1.0 - damp) * recent_mean_log

        noise = rng.normal(0.0, resid_std_log * (noise_level if noise_level is not None else 0.35))
        if rng.random() < (spike_prob if spike_prob is not None else 0.02):
            noise += rng.normal(0.0, resid_std_log * 2.5 * (noise_level if noise_level is not None else 0.35))

        y_pred_log = blended_log + noise
        ylog_series.append(y_pred_log)
        rows.append({"ds": ds_next, "y_hat": float(np.expm1(y_pred_log))})
        ds_last = ds_next

    return pd.DataFrame(rows)

def plot_recent_and_forecast(threat, recent_actual, fcst_df, lookback_hours=48, title_suffix=""):
    fig, ax = plt.subplots(figsize=(12, 5))
    if recent_actual is not None and not recent_actual.empty:
        ax.plot(recent_actual["ds"], recent_actual["y"], label=f"Actual (last {lookback_hours}h)")
    if fcst_df is not None and not fcst_df.empty:
        ax.plot(fcst_df["ds"], fcst_df["y_hat"], label="Forecast", linewidth=2)
    ax.set_title(f"{threat} — Hourly Attacks {title_suffix}")
    ax.set_xlabel("Time"); ax.set_ylabel("Count"); ax.legend()
    return fig

# =========================
# ---- UI  ----------------
# =========================
st.title("🛡️ Predicción de Ataques (Full)")

st.info("**Modo solo sesión** activado: el dataset se mantiene en memoria. Descárgalo si necesitas persistencia.")

# Sidebar controls
with st.sidebar:
    st.header("Hourly features")
    st.caption("Clipping (trim extreme spikes before training)")
    clip_q = st.slider("Recorte superior (cuantil)", 0.90, 0.999, 0.98, 0.001)

    st.markdown("---")
    st.caption("Forecast behavior")
    damp = st.slider("Seasonality strength (0=flat, 1=full)", 0.0, 1.0, 0.60, 0.05)
    noise = st.slider("Noise level", 0.0, 1.0, 0.35, 0.05)
    spike = st.slider("Spike probability", 0.0, 0.2, 0.02, 0.005)

    st.markdown("---")
    p_parq, p_csv = _session_downloads()
    if p_parq:
        st.download_button("⬇️ Download session master.parquet", data=open(p_parq, "rb").read(),
                           file_name="session_master.parquet", mime="application/octet-stream")
    if p_csv:
        st.download_button("⬇️ Download session master.csv", data=open(p_csv, "rb").read(),
                           file_name="session_master.csv", mime="text/csv")

    st.markdown("---")
    if st.button("↻ Clear caches"):
        st.cache_data.clear(); st.cache_resource.clear()
        st.success("Caches cleared. Reloading…"); st.experimental_rerun()

# Ingest controls
st.subheader("1) Agrega Información para Entrenar el Modelo")
uploaded = st.file_uploader("Subir CSV (exportación BDS sin procesar)", type=["csv"])

thin_ingest = st.toggle("🪶 Thin ingest (usar solo columnas relevantes)", True,
                        help="Lee solo columnas necesarias para hora, tipo y firma; reduce memoria.")
chunksize_opt = st.select_slider("Tamaño de chunk", options=[100_000,150_000,200_000,250_000,300_000],
                                 value=250_000, format_func=lambda x: f"{x:,} filas")

# URL path
st.markdown("**O pega un enlace (HTTPS/Dropbox/Google Drive)**")
url_in = st.text_input("URL a CSV (o .gz/.zip con CSV dentro)", placeholder="https://…")
fetch_btn = st.button("Fetch & Merge from URL", use_container_width=True, disabled=not url_in)

def _normalize_direct_download(url: str) -> str:
    url = url.strip()
    if "dropbox.com" in url:
        if "dl=0" in url: url = url.replace("dl=0","dl=1")
        elif "dl=1" not in url and "raw=1" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}dl=1"
    m = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if m:
        fid = m.group(1); url = f"https://drive.google.com/uc?export=download&id={fid}"
    m = re.search(r"drive\.google\.com/open\?id=([^&]+)", url)
    if m:
        fid = m.group(1); url = f"https://drive.google.com/uc?export=download&id={fid}"
    return url

def download_url_to_csv(url: str, base_no_ext: str) -> str:
    import requests, gzip, zipfile
    url = _normalize_direct_download(url)
    raw_path = f"{base_no_ext}__raw.bin"
    csv_path = f"{base_no_ext}.csv"
    with st.status("Fetching from URL…", expanded=True):
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0)); done = 0
            prog = st.progress(0.0)
            with open(raw_path, "wb") as f:
                for ch in r.iter_content(chunk_size=8*1024*1024):
                    if ch:
                        f.write(ch); done += len(ch)
                        if total: prog.progress(min(done/total, 1.0))
            prog.empty()
    # sniff first bytes
    with open(raw_path, "rb") as fh: head = fh.read(4096)
    def looks_html(b: bytes) -> bool:
        h = b.strip().lower(); return h.startswith(b'<!doctype html') or h.startswith(b'<html')
    kind = ("gz" if head[:2]==b'\x1f\x8b' else
            "zip" if head[:4]==b'PK\x03\x04' else
            "html" if looks_html(head) else "csv")
    try:
        if kind == "html":
            st.error("Downloaded HTML (Drive warning page). Provide a direct file link.")
            raise RuntimeError("HTML instead of CSV")
        elif kind == "gz":
            with gzip.open(raw_path, "rb") as src, open(csv_path, "wb") as dst: shutil.copyfileobj(src, dst)
        elif kind == "zip":
            with zipfile.ZipFile(raw_path) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names: raise RuntimeError("ZIP has no CSV")
                with z.open(names[0]) as src, open(csv_path, "wb") as dst: shutil.copyfileobj(src, dst)
        else:
            shutil.move(raw_path, csv_path); raw_path = None
        return csv_path
    finally:
        try:
            if raw_path and os.path.exists(raw_path): os.remove(raw_path)
        except Exception:
            pass

def _handle_ingest(csv_path: str):
    usecols_cb = make_usecols_callable(THIN_INPUT_COLS) if thin_ingest else None
    hourly, rows = process_csv_to_hourly_counts(csv_path, chunksize=chunksize_opt, usecols_filter=usecols_cb)
    base = _session_counts()
    merged = pd.concat([base, hourly], ignore_index=True)
    merged = merged.groupby(["Threat Type","ds"], as_index=False)["y"].sum()
    _set_session_counts(merged)
    # persist session snapshots for download buttons
    _session_downloads()
    return rows, len(merged)

if fetch_btn and url_in:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_no_ext = os.path.join(DATA_DIR, f"remote_{ts}")
    csv_local = download_url_to_csv(url_in, base_no_ext)
    rows, bins = _handle_ingest(csv_local)
    st.success(f"Ingested **{rows:,}** rows → hourly bins in memory: **{bins:,}**")

# File upload path
colA, colB = st.columns([1,1])
with colA:
    default_name = dt.datetime.now().strftime("processed_%Y%m%d_%H%M%S.csv")
    outname = st.text_input("Nombre del archivo procesado (solo para referencia)", value=default_name)
with colB:
    process_btn = st.button("Process & Merge", type="primary", use_container_width=True, disabled=uploaded is None)

if process_btn and uploaded is not None:
    raw_path = os.path.join(DATA_DIR, f"upload_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    uploaded.seek(0)
    with open(raw_path, "wb") as dst:
        shutil.copyfileobj(uploaded, dst, length=16*1024*1024)
    rows, bins = _handle_ingest(raw_path)
    st.success(f"Ingested **{rows:,}** rows → hourly bins in memory: **{bins:,}**")

# Data status
st.divider()
st.subheader("Data Status (session)")
counts = _session_counts()
cov = _coverage_stats_counts(counts)
if cov:
    start, end, n = cov
    st.success(f"Data from **{start}** to **{end}** — hourly rows: **{n:,}**")
    if not counts.empty:
        tt_list = sorted(map(str, counts["Threat Type"].dropna().unique()))
        st.write(f"Threat Types ({len(tt_list)}): " + ", ".join(tt_list[:30]) + (" ..." if len(tt_list) > 30 else ""))
else:
    st.info("No data yet. Upload or fetch to get started.")

# =========================
# ---- TRAIN & FORECAST ----
# =========================
st.divider()
st.subheader("2) Entrenar y Pronosticar")

if counts.empty:
    st.warning("Cargue y procese al menos un CSV primero.")
else:
    threats = sorted(map(str, counts["Threat Type"].dropna().unique()))
    chosen = st.multiselect("Tipos de amenazas", options=threats, default=threats[:1])

    horizon_choice = st.select_slider("Horizonte", options=[7,14,30], value=7, format_func=lambda d: f"{d} días")
    lookback_hours = st.select_slider("Historial real a mostrar", options=[24,48,72,96,120,144,168],
                                      value=48, format_func=lambda h: f"{h//24} días")

    run_btn = st.button("Entrenar y Pronosticar", type="primary", use_container_width=True, disabled=len(chosen)==0)

    if run_btn:
        for threat in chosen:
            with st.spinner(f"Entrenando {threat}…"):
                bundle = train_xgb_for_threat(counts, threat, clip_q=clip_q, test_days=7)

            if bundle is None:
                st.error(f"No hay suficientes datos válidos para **{threat}**.")
                continue

            saved = joblib.load(bundle["model_path"])
            model_bundle = {
                "model": saved["model"],
                "features": saved["features"],
                "cfg": saved["cfg"],
                "resid_std_log": saved.get("resid_std_log", 0.10),
            }

            fcst = forecast_recursive(
                counts, threat, horizon_days=int(horizon_choice),
                model_bundle=model_bundle,
                seasonality_strength=damp, noise_level=noise, spike_prob=spike
            )

            if isinstance(fcst, dict) and fcst.get("insufficient_history"):
                st.warning(
                    f"**{threat}**: historial insuficiente para {horizon_choice}d "
                    f"(necesita ~{fcst['needed']}, disponible {fcst['available']})."
                )
                continue

            g = counts[counts["Threat Type"] == threat].copy()
            g["ds"] = pd.to_datetime(g["ds"], errors="coerce")

            fcst_start = fcst["ds"].min() if fcst is not None and not fcst.empty else g["ds"].max() + pd.Timedelta(hours=1)
            recent_actual = g[(g["ds"] >= fcst_start - pd.Timedelta(hours=lookback_hours)) & (g["ds"] < fcst_start)]

            fig = plot_recent_and_forecast(threat, recent_actual, fcst, lookback_hours=lookback_hours,
                                           title_suffix=f"(+{horizon_choice}d)")
            st.pyplot(fig)

            plot_path = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_choice}d.png")
            fig.savefig(plot_path, dpi=160)
            csv_path  = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_choice}d_fcst.csv")
            fcst.to_csv(csv_path, index=False)

            val = bundle["validation"]
            st.caption(f"Validation (last split): MAE={val['val_mae']:.2f} | RMSE={val['val_rmse']:.2f}")

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download forecast CSV", data=open(csv_path,"rb").read(),
                                   file_name=os.path.basename(csv_path), mime="text/csv")
            with c2:
                st.download_button("⬇️ Download plot PNG", data=open(plot_path,"rb").read(),
                                   file_name=os.path.basename(plot_path), mime="image/png")

st.divider()
st.subheader("Notas & Guardrails")
st.markdown("""
- **Cada fila cuenta**: el roll-up horario se construye directamente del CSV crudo (no se pierde detalle).
- **Recorte de extremos**: ajusta el cuantil superior (por defecto 0.98) para domar picos.
- **Pronóstico**: mezcla del modelo con media reciente (controlada por *Seasonality strength*),
  ruido proporcional al error de validación y picos ocasionales configurables.
- **Modo sesión**: datos en memoria; usa los botones de descarga de la barra lateral.
- **Opcional persistencia**: si desactivas STATELESS_ONLY y llamas _add_part_to_master(...) puedes crear
  un *dataset maestro* en `data/master_parquet/` y leerlo con `_read_master_parquet_unified()`.
""")
st.caption("© Streamlit + XGBoost")
