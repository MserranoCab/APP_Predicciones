# app.py — Cyber Attack Forecasting Tool (Streamlit + XGBoost)
# ------------------------------------------------------------
# Key ideas:
# - Process CSV -> enriched parquet (raw events preserved)
# - Session holds tiny hourly counts for quick UI + list of raw parquet parts
# - Training rebuilds counts from raw parquet parts (uses every row)
# - Forecast blends XGBoost with seasonal damping + stochastic noise

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

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.autolayout": True})

# =========================
# ---- CONFIG / STORAGE ----
# =========================
st.set_page_config(page_title="Cyber Attacks Forecaster", page_icon="🛡️", layout="wide")

DATA_DIR      = "data"
MODELS_DIR    = "models"
PLOTS_DIR     = "plots"
PROCESSED_DIR = "processed"
MASTER_DS_DIR = os.path.join(DATA_DIR, "master_parquet")   # raw-parquet parts live here (session or ephemeral disk)
ENRICH_SUFFIX_PARQUET = "_enriched_raw.parquet"

for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR, PROCESSED_DIR, MASTER_DS_DIR]:
    os.makedirs(d, exist_ok=True)

# ===============================
# ---- Utility / Session state ---
# ===============================
def _normalize_and_uniquify_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(c).strip() for c in df.columns]
    if len(cols) != len(set(cols)):
        seen = {}
        out = []
        for c in cols:
            k = seen.get(c, 0)
            out.append(c if k == 0 else f"{c}.{k}")
            seen[c] = k + 1
        df.columns = out
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

def get_first_series(df: pd.DataFrame, colname: str):
    if colname not in df.columns:
        return None
    obj = df[colname]
    return obj.iloc[:, 0] if isinstance(obj, pd.DataFrame) else obj

def parse_addition_info_column(df: pd.DataFrame) -> pd.DataFrame:
    def parse(info_str):
        if pd.isna(info_str):
            return {}
        pattern = r'type=([\w\[\]\."]+)\s+value=([\w\.\[\]":\-@/\\]+)'
        matches = re.findall(pattern, info_str)
        return {k.strip(): v.strip() for k, v in matches}
    parsed = df["Addition Info"].apply(parse)
    parsed_df = pd.json_normalize(parsed)
    return coalesce_columns(df, parsed_df)

def map_attack_result(df: pd.DataFrame) -> pd.DataFrame:
    result_map = {"1": "Attempted", "2": "Successful", 1: "Attempted", 2: "Successful"}
    s = get_first_series(df, "attack_result")
    if s is None:
        df["attack_result_label"] = np.nan
        return df
    df["attack_result_label"] = s.map(result_map)
    return df

def create_attack_signature(df: pd.DataFrame) -> pd.DataFrame:
    signature_cols = [
        "Threat Name", "Threat Type", "Threat Subtype", "Severity",
        "Source IP", "Destination IP", "Attacker", "Victim"
    ]
    for c in signature_cols:
        if c not in df.columns:
            df[c] = np.nan
    df["attack_signature"] = df[signature_cols].astype(str).agg("|".join, axis=1)
    return df

def _dedupe_events_by_signature_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key = [c for c in ["Attack Start Time", "attack_signature"] if c in df.columns]
    if not key:
        return df
    return (df.sort_values("Attack Start Time")
              .drop_duplicates(subset=key, keep="first"))

# ---------------- Session store ----------------
def _raw_parts() -> list[str]:
    return st.session_state.get("raw_parts", [])

def _add_raw_part(p: str):
    parts = st.session_state.get("raw_parts", [])
    if p not in parts:
        parts.append(p)
    st.session_state["raw_parts"] = parts

def _session_rollup_df() -> pd.DataFrame:
    """Hourly counts kept in-memory for UI speed: ['Threat Type','ds','y']"""
    return st.session_state.get("session_rollup", pd.DataFrame(columns=["Threat Type","ds","y"]))

def _set_session_rollup(df: pd.DataFrame):
    st.session_state["session_rollup"] = df

def _coverage_stats_counts(df: pd.DataFrame):
    if df.empty or "ds" not in df.columns:
        return None
    ds = pd.to_datetime(df["ds"], errors="coerce").dropna()
    if ds.empty:
        return None
    return ds.min().date(), ds.max().date(), len(df)

# ===========================
# ---- CSV -> parquet (raw) --
# ===========================
def make_usecols_callable(keep_cols: set[str]):
    lower_keep = {c.lower() for c in keep_cols}
    def _f(colname: str) -> bool:
        return (colname in keep_cols) or (str(colname).lower() in lower_keep)
    return _f

THIN_INPUT_COLS = {
    "Attack Start Time", "First Seen",    # time
    "Threat Type", "Threat Name", "Threat Subtype", "Severity",
    "Source IP", "Destination IP", "Attacker", "Victim",
    "Addition Info", "attack_result", "direction", "duration",
}

def _ensure_time_column(df: pd.DataFrame) -> pd.DataFrame:
    if "Attack Start Time" not in df.columns:
        if "First Seen" in df.columns:
            df["Attack Start Time"] = pd.to_datetime(df["First Seen"], errors="coerce")
        else:
            raise ValueError(
                "CSV must include 'Attack Start Time' or 'First Seen' column."
            )
    return df

def process_csv_to_hourly_counts(
    csv_path: str,
    chunksize: int = 250_000,
    usecols_filter=None
) -> dict:
    """
    Stream-read CSV -> enrich each chunk -> append to single enriched parquet
    AND build a tiny hourly roll-up in-memory. Returns:
      { 'parquet_path': <str>, 'rows': <int>, 'counts_path': <str> }
    """
    prog = st.progress(0.0, text="Leyendo CSV…")
    rows_done = 0

    ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_base = os.path.join(PROCESSED_DIR, f"processed_{ts_tag}.csv")
    out_parquet = processed_base.replace(".csv", ENRICH_SUFFIX_PARQUET)
    Path(out_parquet).unlink(missing_ok=True)

    addinfo_re = re.compile(r'type=(?P<key>[^ \t]+)\s+value=(?P<val>[^;,\n]+)')
    writer = None
    schema = None
    cols_ref = None
    rollup = None

    # Stream read
    reader = pd.read_csv(
        csv_path,
        low_memory=False,
        chunksize=chunksize,
        usecols=usecols_filter
    )

    for i, df in enumerate(reader):
        df = _normalize_and_uniquify_columns(df)
        df = _ensure_time_column(df)

        if "Addition Info" not in df.columns: df["Addition Info"] = np.nan
        if "attack_result" not in df.columns: df["attack_result"] = np.nan

        # Parse Addition Info (vectorized)
        s = df["Addition Info"].fillna("")
        ext = (
            s.str.extractall(addinfo_re)
            .reset_index()
            .rename(columns={"level_0": "row", "key": "k", "val": "v"})
        )
        if not ext.empty:
            wide = ext.pivot(index="row", columns="k", values="v")
            wide.columns = [str(c).strip() for c in wide.columns]
            wide = wide.reset_index()
            df = (
                df.reset_index(drop=True)
                  .reset_index()
                  .merge(wide, left_on="index", right_on="row", how="left")
                  .drop(columns=["index","row"])
            )

        df = map_attack_result(df)
        df = create_attack_signature(df)

        ts = pd.to_datetime(df["Attack Start Time"], errors="coerce")
        df["Attack Start Time"] = ts
        df["Day"]  = ts.dt.date
        df["Hour"] = ts.dt.hour

        # text columns -> pandas string dtype (Arrow large_string)
        TEXTY = [
            "Addition Info","Threat Name","Threat Type","Threat Subtype",
            "Source IP","Destination IP","Attacker","Victim",
            "direction","Severity","attack_result","attack_result_label"
        ]
        for c in TEXTY:
            if c in df.columns:
                df[c] = df[c].astype("string")

        # Stabilize schema across chunks
        if cols_ref is None:
            cols_ref = list(df.columns)
        else:
            for c in cols_ref:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df.reindex(columns=cols_ref)

        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            schema = table.schema
            writer = pq.ParquetWriter(out_parquet, schema=schema, compression="zstd", use_dictionary=True)
        writer.write_table(table)

        # Update tiny roll-up in memory
        g = pd.DataFrame({
            "Threat Type": df.get("Threat Type", "").astype(str),
            "ds": pd.to_datetime(df["Attack Start Time"], errors="coerce").dt.floor("h")
        }).dropna(subset=["ds"]).groupby(["Threat Type","ds"]).size().reset_index(name="y")

        if rollup is None:
            rollup = g
        else:
            rollup = pd.concat([rollup, g], ignore_index=True)
            if len(rollup) > 250_000:
                rollup = rollup.groupby(["Threat Type","ds"], as_index=False)["y"].sum()

        rows_done += len(df)
        prog.progress(min(0.99, 0.02 + i * 0.02), text=f"Procesadas ~{rows_done:,} filas")

    if writer is not None:
        writer.close()

    # Finalize tiny roll-up
    if rollup is None:
        rollup = pd.DataFrame(columns=["Threat Type","ds","y"])
    else:
        rollup = rollup.groupby(["Threat Type","ds"], as_index=False)["y"].sum()
    rollup["ds"] = pd.to_datetime(rollup["ds"], errors="coerce")
    counts_path = processed_base.replace(".csv", "_hourly_counts.csv")
    rollup.to_csv(counts_path, index=False)

    prog.progress(1.0, text=f"¡Listo! Total procesado: {rows_done:,} filas")
    return {"parquet_path": out_parquet, "rows": rows_done, "counts_path": counts_path}

# ===============================
# ---- Raw parts -> counts -------
# ===============================
def _safe_read_thin_parquet(parquet_path: str, want_cols: list[str]) -> pd.DataFrame:
    try:
        pf = pq.ParquetFile(parquet_path)
        avail = set(pf.schema.names)
        cols = [c for c in want_cols if c in avail]
        if not cols:
            return pd.DataFrame()
        t = pf.read(columns=cols)
        return t.to_pandas()
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def hourly_counts_from_raw_parts(parquet_paths: tuple[str, ...], clip_q: float | None = None) -> pd.DataFrame:
    if not parquet_paths:
        return pd.DataFrame(columns=["Threat Type","ds","y"])
    pieces = []
    for p in parquet_paths:
        df = _safe_read_thin_parquet(p, ["Attack Start Time","Threat Type"])
        if df.empty:
            continue
        ts = pd.to_datetime(df["Attack Start Time"], errors="coerce")
        g = (
            pd.DataFrame({"Threat Type": df.get("Threat Type","").astype(str), "ds": ts.dt.floor("h")})
            .dropna(subset=["ds"])
            .groupby(["Threat Type","ds"])
            .size()
            .reset_index(name="y")
        )
        pieces.append(g)
    if not pieces:
        return pd.DataFrame(columns=["Threat Type","ds","y"])
    out = (pd.concat(pieces, ignore_index=True)
             .groupby(["Threat Type","ds"], as_index=False)["y"].sum()
             .sort_values("ds"))
    if clip_q is not None:
        thr = out["y"].quantile(clip_q)
        out.loc[out["y"] > thr, "y"] = thr
    return out

# ===============================
# ---- Features / Model ----------
# ===============================
WINDOW_CONFIG = {
    "DoS": {"rolling": 3, "lags": [1, 2]},
    "Scan": {"rolling": 6, "lags": [1, 2, 6]},
    "Malicious Flow": {"rolling": 12, "lags": [1, 2, 6]},
    "Vulnerability Attack": {"rolling": 6, "lags": [1, 2, 24]},
    "Attack": {"rolling": 6, "lags": [1, 2]},
    "Malfile": {"rolling": 3, "lags": [1, 2]},
}

def _add_time_features(subset: pd.DataFrame) -> pd.DataFrame:
    subset = subset.copy()
    subset["y_log"] = np.log1p(subset["y"])
    subset["hour"] = subset["ds"].dt.hour
    subset["dayofweek"] = subset["ds"].dt.dayofweek
    subset["is_weekend"] = subset["dayofweek"].isin([5,6]).astype(int)
    subset["is_night"] = subset["hour"].apply(lambda x: 1 if x < 7 or x > 21 else 0)
    subset["weekofyear"] = subset["ds"].dt.isocalendar().week.astype(int)
    subset["time_since_last"] = subset["ds"].diff().dt.total_seconds().div(3600).fillna(0)
    return subset

def _merge_extra_columns(grouped: pd.DataFrame, raw_df: pd.DataFrame | None, threat: str) -> pd.DataFrame:
    """
    If raw_df is None or lacks columns, fill neutral values.
    """
    extra_cols = ["Severity","attack_result_label","direction","duration"]
    merged = grouped.copy()
    if raw_df is None or "Attack Start Time" not in raw_df.columns:
        for c in extra_cols:
            merged[c] = 0
        return merged

    for c in extra_cols:
        if c not in raw_df.columns:
            raw_df[c] = np.nan
    raw_df = raw_df.copy()
    raw_df["ds"] = pd.to_datetime(raw_df["Attack Start Time"], errors="coerce").dt.floor("h")
    extra = raw_df[raw_df["Threat Type"].astype(str) == str(threat)][["ds"] + extra_cols].drop_duplicates(subset="ds")
    merged = pd.merge(grouped, extra, on="ds", how="left")

    le_sev = LabelEncoder()
    le_dir = LabelEncoder()
    merged["Severity"] = le_sev.fit_transform(merged["Severity"].astype(str))
    merged["direction"] = le_dir.fit_transform(merged["direction"].astype(str))
    merged["attack_result_label"] = pd.to_numeric(merged["attack_result_label"], errors="coerce").fillna(0)
    merged["duration"] = pd.to_numeric(merged["duration"], errors="coerce").fillna(0)
    return merged

def _add_lags_rolls(subset: pd.DataFrame, threat: str):
    subset = subset.copy()
    base_cfg = {"rolling": 3, "lags": [1, 2, 6]}
    cfg_src  = WINDOW_CONFIG.get(threat, base_cfg)
    cfg = {"rolling": int(cfg_src.get("rolling", 3)),
           "lags": [int(l) for l in cfg_src.get("lags", [1,2,6]) if int(l) in (1,2,6)]}
    if not cfg["lags"]:
        cfg["lags"] = [1,2,6]
    for lag in cfg["lags"]:
        subset[f"lag{lag}"] = subset["y_log"].shift(lag)
    subset["rolling_mean"] = subset["y_log"].rolling(cfg["rolling"]).mean().shift(1)
    subset["rolling_std"]  = subset["y_log"].rolling(cfg["rolling"]).std().shift(1)
    if "rolling_sum24h" in subset.columns:
        subset = subset.drop(columns=["rolling_sum24h"])
    return subset, cfg

def _enough_history(subset: pd.DataFrame, horizon_hours: int) -> bool:
    return len(subset) >= max(200, int(horizon_hours * 3))

@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    return joblib.load(path)

def train_xgb_for_threat(
    grouped: pd.DataFrame,
    threat: str,
    raw_df_for_merge: pd.DataFrame | None,
    test_days: int = 7,
    clip_q: float = 0.98
):
    if grouped.empty:
        return None

    sub = grouped[grouped["Threat Type"] == threat].copy()
    if len(sub) < 150:
        return None

    if clip_q is not None:
        thr = sub["y"].quantile(clip_q)
        sub.loc[sub["y"] > thr, "y"] = thr

    sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
    sub = _add_time_features(sub)
    sub = _merge_extra_columns(sub, raw_df_for_merge, threat)
    sub, cfg = _add_lags_rolls(sub, threat)

    feature_cols = [
        "hour","dayofweek","is_weekend","is_night","weekofyear","time_since_last",
        "Severity","attack_result_label","direction","duration"
    ]
    feature_cols += [c for c in ["lag1","lag2","lag6","lag24"] if c in sub.columns]
    for c in ["rolling_mean","rolling_std","rolling_sum24h"]:
        if c in sub.columns:
            feature_cols.append(c)

    sub = sub.dropna(subset=feature_cols)
    if sub.empty:
        return None

    cutoff = sub["ds"].max() - pd.Timedelta(days=test_days)
    train = sub[sub["ds"] <= cutoff]
    test  = sub[sub["ds"]  > cutoff]
    if len(train) < 50 or len(test) < 20:
        return None

    X_full, y_full = train[feature_cols], train["y_log"]
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        objective="reg:squarederror", n_jobs=4
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred_log = model.predict(X_val)
    val_mae  = float(np.mean(np.abs(np.expm1(val_pred_log) - np.expm1(y_val))))
    val_rmse = float(np.sqrt(np.mean((np.expm1(val_pred_log) - np.expm1(y_val))**2)))
    resid_std_log = float(np.std(y_val - val_pred_log))

    mpath = os.path.join(MODELS_DIR, f"xgb_{re.sub('[^A-Za-z0-9]+','_', threat)}.joblib")
    joblib.dump({"model": model, "features": feature_cols, "cfg": cfg, "resid_std_log": resid_std_log}, mpath)

    return {
        "model_path": mpath,
        "features": feature_cols,
        "cfg": cfg,
        "train": train,
        "test": test,
        "validation": {"val_mae": val_mae, "val_rmse": val_rmse, "resid_std_log": resid_std_log},
    }

def forecast_recursive(
    grouped: pd.DataFrame,
    threat: str,
    horizon_days: int,
    model_bundle: dict,
    seasonality_strength: float = 0.6,
    noise_level: float = 0.35,
    spike_prob: float = 0.02,
    seed: int = 1234
) -> pd.DataFrame | dict | None:
    rng = np.random.default_rng(seed)

    sub = grouped[grouped["Threat Type"] == threat].copy()
    if sub.empty:
        return None
    sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
    sub = _add_time_features(sub)
    sub, _ = _add_lags_rolls(sub, threat)

    model = model_bundle["model"]
    features = model_bundle["features"]
    resid_std_log = float(model_bundle.get("resid_std_log", 0.10))

    history = sub.dropna(subset=features).copy().sort_values("ds")
    if history.empty:
        return None

    horizon_hours = int(horizon_days * 24)
    if not _enough_history(history, horizon_hours):
        return {"insufficient_history": True, "needed": max(200, horizon_hours * 3), "available": len(history)}

    max_lag = max([int(x[3:]) for x in features if x.startswith("lag")] + [1])
    buffer_points = max(max_lag + 24, 30)
    hist_tail = history.tail(buffer_points).copy()
    ylog_series = hist_tail["y_log"].tolist()
    ds_last = history["ds"].max()

    rows = []
    for _ in range(horizon_hours):
        ds_next = ds_last + pd.Timedelta(hours=1)
        hour = ds_next.hour
        dayofweek = ds_next.dayofweek
        is_weekend = int(dayofweek in [5,6])
        is_night = 1 if hour < 7 or hour > 21 else 0
        weekofyear = int(pd.Timestamp(ds_next).isocalendar().week)

        row = {
            "ds": ds_next,
            "hour": hour, "dayofweek": dayofweek, "is_weekend": is_weekend,
            "is_night": is_night, "weekofyear": weekofyear, "time_since_last": 1.0,
        }
        for c in features:
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

        if len(ylog_series) >= 24:
            row["rolling_sum24h"] = float(pd.Series(ylog_series[-24:]).sum())
        else:
            row["rolling_sum24h"] = np.nan

        X_row = pd.DataFrame([row]).dropna(subset=features)
        base_pred_log = ylog_series[-1] if X_row.empty else float(model.predict(X_row[features])[0])

        recent_mean_log = float(pd.Series(ylog_series[-24:]).mean()) if len(ylog_series) >= 24 else float(np.mean(ylog_series))
        blended_log = seasonality_strength * base_pred_log + (1.0 - seasonality_strength) * recent_mean_log

        noise = rng.normal(0.0, resid_std_log * noise_level)
        if rng.random() < spike_prob:
            noise += rng.normal(0.0, resid_std_log * 2.5 * noise_level)

        y_pred_log = blended_log + noise
        ylog_series.append(y_pred_log)
        rows.append({"ds": ds_next, "y_hat": float(np.expm1(y_pred_log))})
        ds_last = ds_next

    return pd.DataFrame(rows)

def plot_recent_and_forecast(threat: str, recent_actual: pd.DataFrame, fcst_df: pd.DataFrame, lookback_hours: int = 48, title_suffix: str = "", cap_quantile: float = 0.995):
    fig, ax = plt.subplots(figsize=(12,5))
    vals = []
    if recent_actual is not None and not recent_actual.empty:
        ax.plot(recent_actual["ds"], recent_actual["y"], label=f"Actual (last {lookback_hours}h)")
        vals.append(recent_actual["y"].values)
    if fcst_df is not None and not fcst_df.empty:
        ax.plot(fcst_df["ds"], fcst_df["y_hat"], label="Forecast", linewidth=2)
        vals.append(fcst_df["y_hat"].values)
    if vals:
        vmax = float(np.quantile(np.concatenate(vals), cap_quantile))
        ax.set_ylim(0, max(1.0, vmax) * 1.1)
    ax.set_title(f"{threat} — Hourly Attacks {title_suffix}")
    ax.set_xlabel("Time"); ax.set_ylabel("Count"); ax.legend()
    return fig

# ===========================
# ---- URL Download helper ---
# ===========================
def download_url_to_csv(url: str, base_path_no_ext: str) -> str:
    import requests, gzip, zipfile
    def _direct(u: str) -> str:
        u = u.strip()
        if "dropbox.com" in u and "dl=1" not in u:
            u = u.replace("dl=0", "dl=1") if "dl=0" in u else (u + ("&" if "?" in u else "?") + "dl=1")
        m = re.search(r"drive\.google\.com/file/d/([^/]+)", u)
        if m:
            u = f"https://drive.google.com/uc?export=download&id={m.group(1)}"
        m = re.search(r"drive\.google\.com/open\?id=([^&]+)", u)
        if m:
            u = f"https://drive.google.com/uc?export=download&id={m.group(1)}"
        return u

    url = _direct(url)
    raw_path = f"{base_path_no_ext}__raw.bin"
    csv_path = f"{base_path_no_ext}.csv"

    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        prog = st.progress(0.0)
        with open(raw_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8*1024*1024):
                if chunk:
                    f.write(chunk); done += len(chunk)
                    if total:
                        prog.progress(min(done/total, 1.0))
        prog.empty()

    with open(raw_path, "rb") as fh:
        head = fh.read(4)
    try:
        if head[:2] == b"\x1f\x8b":  # gz
            with gzip.open(raw_path, "rb") as src, open(csv_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
        elif head[:4] == b"PK\x03\x04":  # zip
            with zipfile.ZipFile(raw_path) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise RuntimeError("ZIP has no CSV inside.")
                with z.open(names[0]) as src, open(csv_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        else:
            shutil.move(raw_path, csv_path)
            raw_path = None
        return csv_path
    finally:
        try:
            if raw_path and os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception:
            pass

# ===========================
# ---- STREAMLIT UI ----------
# ===========================
st.title("🛡️ Predicción de Ataques")
st.caption("Subir → Procesar → Entrenar → Predecir")

# ---- Sidebar: Data Status ----
with st.sidebar:
    st.header("📦 Data Status")
    roll = _session_rollup_df()
    cov = _coverage_stats_counts(roll)
    if cov:
        start, end, n = cov
        st.success(f"Data from **{start}** to **{end}**  \nRows: **{n:,}**")
        if not roll.empty:
            tt_list = sorted(map(str, roll["Threat Type"].dropna().unique()))
            st.write(f"Threat Types ({len(tt_list)}):")
            st.write(", ".join(tt_list[:30]) + (" ..." if len(tt_list) > 30 else ""))

        # downloads of current session roll-up
        if not roll.empty:
            parq_path = os.path.join(DATA_DIR, "session_master.parquet")
            csv_path  = os.path.join(DATA_DIR, "session_master.csv")
            pq.write_table(pa.Table.from_pandas(roll, preserve_index=False), parq_path, compression="zstd")
            roll.to_csv(csv_path, index=False)
            st.download_button("⬇️ Download session master.parquet", data=open(parq_path,"rb").read(), file_name="session_master.parquet", mime="application/octet-stream")
            st.download_button("⬇️ Download session master.csv",      data=open(csv_path,"rb").read(),  file_name="session_master.csv",      mime="text/csv")
    else:
        st.info("No data yet. Upload or fetch to get started.")

    st.markdown("---")
    if st.button("↻ Clear caches"):
        st.cache_data.clear(); st.cache_resource.clear()
        st.success("Caches cleared. Reloading…")
        st.experimental_rerun()

# ---- Upload / URL ingest ----
st.subheader("1) Agrega Información para Entrenar el Modelo")

thin_ingest = st.toggle("🪶 Thin ingest (leer solo columnas relevantes)", value=True)
usecols_cb = make_usecols_callable(THIN_INPUT_COLS) if thin_ingest else None

col_u1, col_u2 = st.columns([2,1])
with col_u1:
    url_in = st.text_input("URL (CSV / .gz / .zip con CSV adentro)", placeholder="https://…")
with col_u2:
    fetch_btn = st.button("Fetch & Merge from URL", use_container_width=True, disabled=not url_in)

uploaded = st.file_uploader("…o sube un CSV", type=["csv"])
process_btn = st.button("Process & Merge", type="primary", use_container_width=True, disabled=uploaded is None)

def _handle_ingest(csv_path: str):
    with st.status("Reading & enriching (session mode)…", expanded=True) as s:
        result = process_csv_to_hourly_counts(csv_path, chunksize=250_000, usecols_filter=usecols_cb)
        s.write(f"¡Listo! Total procesado: {result['rows']:,} filas")
        # merge tiny roll-up into session
        counts_df = pd.read_csv(result["counts_path"])
        counts_df["ds"] = pd.to_datetime(counts_df["ds"], errors="coerce")
        base = _session_rollup_df()
        merged = pd.concat([base, counts_df], ignore_index=True)
        merged = merged.groupby(["Threat Type","ds"], as_index=False)["y"].sum()
        _set_session_rollup(merged)
        # register raw parquet part for future training
        _add_raw_part(result["parquet_path"])
        s.update(label="Merge complete ✅", state="complete")

if fetch_btn:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_no_ext = os.path.join(DATA_DIR, f"remote_{ts}")
    csv_local = download_url_to_csv(url_in, base_no_ext)
    _handle_ingest(csv_local)

if process_btn and uploaded is not None:
    raw_path = os.path.join(DATA_DIR, f"upload_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    uploaded.seek(0)
    with open(raw_path, "wb") as dst:
        shutil.copyfileobj(uploaded, dst, length=16*1024*1024)
    _handle_ingest(raw_path)

st.divider()

# ---- Hourly feature controls ----
st.sidebar.subheader("Hourly features")
clip_q = st.sidebar.slider("Recorte superior (cuantil)", min_value=0.90, max_value=0.999, value=0.98, step=0.005)
use_raw_for_training = st.sidebar.toggle("Build from raw rows (uses every row)", value=True,
                                         help="Rebuild hourly counts from all enriched parquet parts for training.")
cap_q = st.sidebar.slider("Plot cap quantile (y-axis)", min_value=0.90, max_value=0.999, value=0.995, step=0.001)

st.sidebar.subheader("Forecast behavior")
seasonality_strength = st.sidebar.slider("Seasonality strength (0=flat, 1=full)", 0.0, 1.0, 0.60, 0.01)
noise_level = st.sidebar.slider("Noise level", 0.0, 1.0, 0.35, 0.01)
spike_prob = st.sidebar.slider("Spike probability", 0.0, 0.2, 0.02, 0.005)

# ---- Training & Forecast ----
st.subheader("2) Entrenar el modelo y generar predicciones")

rollup = _session_rollup_df()
if rollup.empty:
    st.warning("Cargue y procese al menos un CSV primero.")
else:
    threats = sorted(map(str, rollup["Threat Type"].dropna().unique()))
    chosen = st.multiselect("Elija los tipos de amenazas", options=threats, default=threats[:1])

    horizon_days = st.select_slider("Horizonte", options=[7, 14, 30], value=7, format_func=lambda d: f"{d} días")
    lookback_hours = st.select_slider("Historial real a mostrar", options=[24,48,72,96,120,144,168], value=48,
                                      format_func=lambda h: f"{h//24} días")

    run_btn = st.button("Entrenar y Pronosticar", type="primary", use_container_width=True, disabled=len(chosen)==0)

    if run_btn:
        # Build grouped counts for training
        if use_raw_for_training and _raw_parts():
            grouped_all = hourly_counts_from_raw_parts(tuple(_raw_parts()), clip_q=None)
        else:
            grouped_all = rollup.copy()

        for threat in chosen:
            with st.spinner(f"Entrenando {threat}…"):
                # raw_df_for_merge=None keeps auxiliary cols neutral; that’s fine
                bundle = train_xgb_for_threat(grouped_all, threat, raw_df_for_merge=None, test_days=7, clip_q=clip_q)

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

            fcst = forecast_recursive(
                grouped_all, threat, horizon_days=int(horizon_days), model_bundle=model_bundle,
                seasonality_strength=seasonality_strength, noise_level=noise_level, spike_prob=spike_prob
            )
            if isinstance(fcst, dict) and fcst.get("insufficient_history"):
                st.warning(
                    f"**{threat}**: Historial insuficiente para {horizon_days}d "
                    f"(necesario ~{fcst['needed']}, disponible {fcst['available']})."
                )
                continue

            grouped = rollup[rollup["Threat Type"] == threat].copy()
            grouped["ds"] = pd.to_datetime(grouped["ds"], errors="coerce")
            fcst_start = fcst["ds"].min() if fcst is not None and not fcst.empty else grouped["ds"].max() + pd.Timedelta(hours=1)
            recent_actual = grouped[(grouped["ds"] >= fcst_start - pd.Timedelta(hours=lookback_hours)) & (grouped["ds"] < fcst_start)]

            fig = plot_recent_and_forecast(threat, recent_actual, fcst, lookback_hours=lookback_hours, title_suffix=f"(+{horizon_days}d)", cap_quantile=cap_q)
            st.pyplot(fig)

            plot_path = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_days}d.png")
            fig.savefig(plot_path, dpi=160)
            csv_path = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_days}d_fcst.csv")
            fcst.to_csv(csv_path, index=False)

            val = bundle["validation"]
            st.caption(f"Validation (last split): MAE={val['val_mae']:.2f} | RMSE={val['val_rmse']:.2f}")

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇️ Download forecast CSV", data=open(csv_path, "rb").read(),
                                   file_name=os.path.basename(csv_path), mime="text/csv")
            with c2:
                st.download_button("⬇️ Download plot PNG", data=open(plot_path, "rb").read(),
                                   file_name=os.path.basename(plot_path), mime="image/png")

st.divider()
st.subheader("Notas & Guardrails")
st.markdown("""
- **Training from raw**: enable *Build from raw rows (uses every row)* to rebuild counts from all enriched parquet parts.
- **Clipping**: set *Recorte superior (cuantil)* to tame extreme spikes before training.
- **Plot cap**: *Plot cap quantile* prevents the forecast from looking like 0 when there’s a recent huge spike.
- **Session mode**: the tiny hourly dataset is kept in memory; download it if you need to persist.
""")
st.caption("© Streamlit + XGBoost")
