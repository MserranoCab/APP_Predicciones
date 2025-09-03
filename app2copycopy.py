# app.py — Cyber Attack Forecasting Tool (Streamlit + XGBoost)
# Stable schema appends, safe "Addition Info" parsing, and XGB hourly forecasts.

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
import gzip
import zipfile
from pathlib import Path
from typing import Optional

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

# =============== CONFIG ===============
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.autolayout": True})

DATA_DIR = "data"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
PROCESSED_DIR = "processed"
SEEDS_DIR = "seeds"
MASTER_DS_DIR = os.path.join(DATA_DIR, "master_parquet")
ENRICH_SUFFIX_PARQUET = "_enriched_raw.parquet"

MASTER_CSV = os.path.join(DATA_DIR, "master.csv")
SEED_FLAG = os.path.join(DATA_DIR, ".seeded")

for d in [DATA_DIR, MODELS_DIR, PLOTS_DIR, PROCESSED_DIR, SEEDS_DIR, MASTER_DS_DIR]:
    os.makedirs(d, exist_ok=True)

st.set_page_config(page_title="Cyber Attacks Forecaster", page_icon="🛡️", layout="wide")


# =============== SMALL UTILS ===============
def coalesce_columns(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    for c in extra.columns:
        if c in base.columns:
            base[c] = base[c].where(base[c].notna(), extra[c])
        else:
            base[c] = extra[c]
    return base

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

def get_first_series(df: pd.DataFrame, colname: str):
    if colname not in df.columns:
        return None
    obj = df[colname]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


# =============== DEDUPES ===============
def _dedupe_events_by_signature_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key = [c for c in ["Attack Start Time", "attack_signature"] if c in df.columns]
    if not key:
        return df
    return (df.sort_values("Attack Start Time")
              .drop_duplicates(subset=key, keep="first"))

def _dedupe_master(df: pd.DataFrame) -> pd.DataFrame:
    if "Attack Start Time" in df.columns:
        df = df.sort_values("Attack Start Time")
    return df.drop_duplicates(keep="first")


# =============== ADDITION INFO / LABELS ===============
addinfo_re = re.compile(r'type=(?P<key>[^ \t]+)\s+value=(?P<val>[^;,\n]+)')

def map_attack_result(df: pd.DataFrame) -> pd.DataFrame:
    result_map = {"1": "Attempted", "2": "Successful", 1: "Attempted", 2: "Successful"}
    s = get_first_series(df, "attack_result")
    if s is None:
        df = df.copy()
        df["attack_result_label"] = np.nan
        return df
    df = df.copy()
    df["attack_result_label"] = s.map(result_map)
    return df

def create_attack_signature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    signature_cols = [
        'Threat Name', 'Threat Type', 'Threat Subtype', 'Severity',
        'Source IP', 'Destination IP', 'Attacker', 'Victim'
    ]
    for c in signature_cols:
        if c not in df.columns:
            df[c] = np.nan
    df['attack_signature'] = df[signature_cols].astype(str).agg('|'.join, axis=1)
    return df

def _vectorized_parse(df_chunk: pd.DataFrame) -> pd.DataFrame:
    # Ensure column exists
    if "Addition Info" not in df_chunk.columns:
        df_chunk = df_chunk.copy()
        df_chunk["Addition Info"] = np.nan

    s = df_chunk["Addition Info"].fillna("")
    ext = (
        s.str.extractall(addinfo_re)
         .reset_index()
         .rename(columns={"level_0": "row", "key": "k", "val": "v"})
    )
    if ext.empty:
        return df_chunk

    wide = ext.pivot(index="row", columns="k", values="v").reset_index()
    wide.columns = [str(c).strip() for c in wide.columns]
    wide = wide.rename(columns={c: (f"ai_{c}" if c != "row" else c) for c in wide.columns})
    out = (
        df_chunk.reset_index(drop=True)
                .reset_index()
                .merge(wide, left_on="index", right_on="row", how="left")
                .drop(columns=["index", "row"])
    )
    # Coalesce ai_ columns into existing base names if clashes
    for c in list(df_chunk.columns):
        ac = f"ai_{c}"
        if ac in out.columns:
            out[c] = out[c].where(out[c].notna(), out[ac])
            out.drop(columns=[ac], inplace=True)
    # Ensure remaining ai_* are string
    for c in [c for c in out.columns if c.startswith("ai_")]:
        out[c] = out[c].astype("string")
    return out


# =============== PARQUET SCHEMA UNION HELPERS ===============
def _resolve_common_type(name: str, old_t: Optional[pa.DataType], new_t: Optional[pa.DataType]) -> pa.DataType:
    # Pin known columns; fallback to large_string on conflict.
    if name == "Attack Start Time":
        return pa.timestamp("ns")
    if name == "Hour":
        return pa.int64()
    if name == "Day":
        return pa.large_string()  # consistent across chunks
    # default: if conflict, use string
    if old_t is None:
        return new_t or pa.large_string()
    if new_t is None:
        return old_t
    return old_t if old_t == new_t else pa.large_string()

def _cast_to_schema(tbl: pa.Table, schema_map: dict) -> pa.Table:
    arrays, names = [], []
    n = tbl.num_rows
    for name, typ in schema_map.items():
        if name in tbl.column_names:
            col = tbl[name]
            if col.type != typ:
                try:
                    arr = pc.cast(col, typ)
                except Exception:
                    arr = pa.array(col.to_pandas().astype(str), type=pa.large_string())
            else:
                arr = col
        else:
            arr = pa.nulls(n, type=typ)
        arrays.append(arr)
        names.append(name)
    return pa.table(arrays, names=names)


# =============== MASTER IO ===============
def _add_enriched_parquet_to_master(parquet_path: str):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(MASTER_DS_DIR, f"part-{ts}.parquet")
    shutil.copyfile(parquet_path, dst)
    return dst

@st.cache_data(show_spinner=False)
def read_master_cached():
    return _read_master()

def _read_master_parquet() -> pd.DataFrame:
    part_paths = sorted(glob.glob(os.path.join(MASTER_DS_DIR, "*.parquet")))
    if not part_paths:
        return pd.DataFrame()

    qdir = os.path.join(MASTER_DS_DIR, "_quarantine")
    os.makedirs(qdir, exist_ok=True)

    # Determine union target schema by scanning parts
    observed = {}
    good_paths = []
    for p in part_paths:
        try:
            t = pq.read_table(p)
            for f in t.schema:
                observed.setdefault(f.name, f.type)
            good_paths.append(p)
        except Exception as e:
            try:
                shutil.move(p, os.path.join(qdir, os.path.basename(p)))
            except Exception:
                pass
            st.warning(f"Quarantined bad parquet: {os.path.basename(p)} — {e}")

    if not good_paths:
        return pd.DataFrame()

    # Construct a relaxed target schema (string for unknowns, pin times)
    names = sorted(observed.keys())
    schema_map = {}
    for name in names:
        typ = observed[name]
        schema_map[name] = _resolve_common_type(name, typ, typ)

    tables = []
    for p in good_paths:
        try:
            tbl = pq.read_table(p)
            tbl_cast = _cast_to_schema(tbl, schema_map)
            tables.append(tbl_cast)
        except Exception as e:
            try:
                shutil.move(p, os.path.join(qdir, os.path.basename(p)))
            except Exception:
                pass
            st.warning(f"Quarantined during cast: {os.path.basename(p)} — {e}")

    if not tables:
        return pd.DataFrame()

    table = pa.concat_tables(tables, promote=True)
    df = table.to_pandas()

    # Final post-cast enhancements
    if "Attack Start Time" in df.columns:
        df["Attack Start Time"] = pd.to_datetime(df["Attack Start Time"], errors="coerce")
        df["Day"] = df["Attack Start Time"].dt.date.astype(str)
        df["Hour"] = df["Attack Start Time"].dt.hour
    df = _dedupe_events_by_signature_time(df)
    return df

def _read_master() -> pd.DataFrame:
    # One-time migrate legacy CSV if present
    if os.path.exists(MASTER_CSV):
        df_legacy = pd.read_csv(MASTER_CSV, low_memory=False, parse_dates=["Attack Start Time"])
        tmp = os.path.join(DATA_DIR, "legacy_master.parquet")
        pq.write_table(pa.Table.from_pandas(df_legacy, preserve_index=False), tmp, compression="zstd")
        _add_enriched_parquet_to_master(tmp)
        os.remove(MASTER_CSV)
        Path(tmp).unlink(missing_ok=True)
    return _read_master_parquet()

def _write_master(df: pd.DataFrame):
    tmp = os.path.join(DATA_DIR, f"snapshot_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet")
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), tmp, compression="zstd")
    _add_enriched_parquet_to_master(tmp)
    Path(tmp).unlink(missing_ok=True)

def _update_master_with_processed(enriched_info):
    parquet_path = enriched_info["parquet_path"] if isinstance(enriched_info, dict) else enriched_info
    _add_enriched_parquet_to_master(parquet_path)
    df = _read_master_parquet()
    return df

def _coverage_stats(df: pd.DataFrame):
    if df.empty or "Attack Start Time" not in df.columns:
        return None
    ts = df["Attack Start Time"].dropna()
    if ts.empty:
        return None
    return ts.min(), ts.max(), len(df)


# =============== SEED BOOTSTRAP ===============
def _discover_seed_paths() -> list:
    candidates = []
    env_paths = os.environ.get("CYBER_SEED_CVS", "")
    if env_paths:
        candidates.extend([p.strip() for p in env_paths.split(",") if p.strip()])
    candidates.extend(glob.glob(os.path.join(SEEDS_DIR, "*.csv")))
    for p in ["BDS_BIG_2MONTHS.csv", "BDS_UNIFICADO.csv", "BDS1.csv", "/mnt/data/BDS1.csv"]:
        if os.path.exists(p):
            candidates.append(p)
    seen, uniq = set(), []
    for c in candidates:
        if c not in seen and os.path.exists(c):
            seen.add(c); uniq.append(c)
    return uniq

def _merge_or_process_seed(path: str) -> pd.DataFrame:
    df_try = pd.read_csv(path, low_memory=False)
    if "attack_signature" in df_try.columns and ("Day" in df_try.columns or "First Seen" in df_try.columns):
        if "Attack Start Time" not in df_try.columns and "First Seen" in df_try.columns:
            df_try["Attack Start Time"] = pd.to_datetime(df_try["First Seen"], errors="coerce")
        return df_try
    outp = os.path.join(PROCESSED_DIR, f"seed_{os.path.basename(path)}")
    return process_log_csv(path, outp)

def _bootstrap_seed_data():
    if os.path.exists(SEED_FLAG):
        return
    paths = _discover_seed_paths()
    if not paths:
        return
    frames = []
    for p in paths:
        try:
            frames.append(_merge_or_process_seed(p))
        except Exception as e:
            st.warning(f"Seed load failed for {p}: {e}")
    if frames:
        master = pd.concat(frames, ignore_index=True)
        master = _dedupe_master(master)
        _write_master(master)
        with open(SEED_FLAG, "w") as f:
            f.write(dt.datetime.now().isoformat())


# =============== CSV -> ENRICH -> PARQUET (non-chunk) ===============
def calculate_recurrence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['Attack Start Time'] = pd.to_datetime(df['Attack Start Time'], errors="coerce")
    grouped = df.groupby('attack_signature')
    first_rows = grouped.first().reset_index()
    agg_info = grouped['Attack Start Time'].agg(['min', 'max', 'count']).reset_index()
    agg_info.columns = ['attack_signature', 'First Seen', 'Last Seen', 'Recurrence Index']
    agg_info['Time Frame (hrs)'] = (agg_info['Last Seen'] - agg_info['First Seen']).dt.total_seconds() / 3600

    def _avg_gap(row):
        if row['Recurrence Index'] > 1 and row['Time Frame (hrs)'] > 0.01:
            return row['Time Frame (hrs)'] / (row['Recurrence Index'] - 1)
        return None

    agg_info['Avg Time Between Events (hrs)'] = agg_info.apply(_avg_gap, axis=1)
    return pd.merge(first_rows, agg_info, on='attack_signature')

def process_log_csv(input_path: str, output_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path, low_memory=False)
    df = _normalize_and_uniquify_columns(df)
    if "Addition Info" not in df.columns:
        df["Addition Info"] = np.nan
    if "attack_result" not in df.columns:
        df["attack_result"] = np.nan
    if "Attack Start Time" not in df.columns:
        if "First Seen" in df.columns:
            df["Attack Start Time"] = pd.to_datetime(df["First Seen"], errors="coerce")
        else:
            raise ValueError("CSV must include 'Attack Start Time' column.")

    df = _vectorized_parse(df)
    df = map_attack_result(df)
    df = create_attack_signature(df)

    ts = pd.to_datetime(df["Attack Start Time"], errors="coerce")
    df["Attack Start Time"] = ts
    df["Day"] = ts.dt.date.astype(str)
    df["Hour"] = ts.dt.hour

    raw_out = output_path.replace(".csv", "_enriched_raw.csv")
    df_out = df.copy()
    df_out["Attack Start Time"] = df_out["Attack Start Time"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_out.to_csv(raw_out, index=False)

    df_final = calculate_recurrence(df)
    df_final = df_final.merge(df[["attack_signature", "Day", "Hour"]], on="attack_signature", how="left")
    df_final.to_csv(output_path, index=False)
    return df


# =============== CSV -> ENRICH -> PARQUET (chunked) ===============
def process_log_csv_with_progress(input_path: str, output_path: str, chunksize: int = 250_000, fast_mode: bool = True):
    prog = st.progress(0.0, text="Leyendo CSV…")
    rows_done = 0
    out_parquet = output_path.replace(".csv", ENRICH_SUFFIX_PARQUET)
    Path(out_parquet).unlink(missing_ok=True)

    TEXTY_COLS = [
        "Addition Info", "Threat Name", "Threat Type", "Threat Subtype",
        "Source IP", "Destination IP", "Attacker", "Victim",
        "direction", "Severity", "attack_result", "attack_result_label"
    ]

    for i, df in enumerate(pd.read_csv(input_path, low_memory=False, chunksize=chunksize)):
        df = _normalize_and_uniquify_columns(df)

        # ensure required columns exist
        if "Addition Info" not in df.columns: df["Addition Info"] = np.nan
        if "attack_result" not in df.columns: df["attack_result"] = np.nan
        if "Attack Start Time" not in df.columns:
            if "First Seen" in df.columns:
                df["Attack Start Time"] = pd.to_datetime(df["First Seen"], errors="coerce")
            else:
                raise ValueError("CSV must include 'Attack Start Time' column.")

        # parse + map + signature
        df = _vectorized_parse(df)
        for c in TEXTY_COLS:
            if c in df.columns:
                df[c] = df[c].astype("string")
        df = map_attack_result(df)
        df = create_attack_signature(df)

        ts = pd.to_datetime(df["Attack Start Time"], errors="coerce")
        df["Attack Start Time"] = ts
        df["Day"] = ts.dt.strftime("%Y-%m-%d")
        df["Hour"] = ts.dt.hour

        # write/append with schema union-cast
        if not Path(out_parquet).exists():
            table = pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(table, out_parquet, compression="zstd", use_dictionary=True)
        else:
            old = pq.read_table(out_parquet)
            new = pa.Table.from_pandas(df, preserve_index=False)

            all_names = sorted(set(old.column_names) | set(new.column_names))
            schema_map = {}
            for name in all_names:
                old_t = old.schema.field(name).type if name in old.column_names else None
                new_t = new.schema.field(name).type if name in new.column_names else None
                schema_map[name] = _resolve_common_type(name, old_t, new_t)

            old_cast = _cast_to_schema(old, schema_map)
            new_cast = _cast_to_schema(new, schema_map)

            merged = pa.concat_tables([old_cast, new_cast], promote=True)
            pq.write_table(merged, out_parquet, compression="zstd", use_dictionary=True)

        rows_done += len(df)
        prog.progress(min(0.99, 0.02 + i * 0.02), text=f"Procesadas ~{rows_done:,} filas")

    if not fast_mode:
        df_all = pq.read_table(out_parquet).to_pandas()
        df_final = calculate_recurrence(df_all)
        df_final = df_final.merge(df_all[["attack_signature", "Day", "Hour"]], on="attack_signature", how="left")
        df_final.to_csv(output_path, index=False)
    else:
        pd.DataFrame({"note": ["Fast mode: resumen omitido."]}).to_csv(output_path, index=False)

    prog.progress(1.0, text=f"¡Listo! Total procesado: {rows_done:,} filas")
    return {"parquet_path": out_parquet, "rows": rows_done, "fast_mode": fast_mode}


# =============== FEATURES / MODELING ===============
WINDOW_CONFIG = {
    'DoS': {'rolling': 3, 'lags': [1, 2]},
    'Scan': {'rolling': 6, 'lags': [1, 2, 6]},
    'Malicious Flow': {'rolling': 12, 'lags': [1, 2, 6]},
    'Vulnerability Attack': {'rolling': 6, 'lags': [1, 2, 24]},
    'Attack': {'rolling': 6, 'lags': [1, 2]},
    'Malfile': {'rolling': 3, 'lags': [1, 2]},
}

def build_hourly_counts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.copy()
    tmp["ds"] = pd.to_datetime(tmp["Attack Start Time"], errors="coerce").dt.floor("h")
    tmp["Threat Type"] = tmp.get("Threat Type", "").astype(str)
    grouped = tmp.groupby(["Threat Type", "ds"]).size().reset_index(name="y")
    return grouped

@st.cache_data(show_spinner=False)
def hourly_counts_cached(df: pd.DataFrame):
    return build_hourly_counts(df)

def _merge_extra_columns(grouped: pd.DataFrame, raw_df: pd.DataFrame, threat: str) -> pd.DataFrame:
    extra_cols = ['Severity', 'attack_result_label', 'direction', 'duration']
    raw_df = raw_df.copy()
    for c in extra_cols:
        if c not in raw_df.columns:
            raw_df[c] = np.nan
    raw_df["ds"] = pd.to_datetime(raw_df["Attack Start Time"], errors="coerce").dt.floor("h")
    extra_data = raw_df[raw_df['Threat Type'].astype(str) == str(threat)][['ds'] + extra_cols].drop_duplicates(subset='ds')
    merged = pd.merge(grouped, extra_data, on='ds', how='left')

    # Encoders expect strings
    le_sev = LabelEncoder()
    le_dir = LabelEncoder()
    merged['Severity'] = le_sev.fit_transform(merged['Severity'].astype(str))
    merged['direction'] = le_dir.fit_transform(merged['direction'].astype(str))
    merged['attack_result_label'] = pd.to_numeric(merged['attack_result_label'], errors='coerce').fillna(0)
    merged['duration'] = pd.to_numeric(merged['duration'], errors='coerce').fillna(0)
    return merged

def _add_time_features(subset: pd.DataFrame) -> pd.DataFrame:
    subset = subset.copy()
    subset['y_log'] = np.log1p(subset['y'])
    subset['hour'] = subset['ds'].dt.hour
    subset['dayofweek'] = subset['ds'].dt.dayofweek
    subset['is_weekend'] = subset['dayofweek'].isin([5, 6]).astype(int)
    subset['is_night'] = subset['hour'].apply(lambda x: 1 if x < 7 or x > 21 else 0)
    subset['weekofyear'] = subset['ds'].dt.isocalendar().week.astype(int)
    subset['time_since_last'] = subset['ds'].diff().dt.total_seconds().div(3600).fillna(0)
    return subset

def _add_lags_rolls(subset: pd.DataFrame, threat: str):
    subset = subset.copy()
    base_cfg = {'rolling': 3, 'lags': [1, 2, 6]}
    cfg_src = WINDOW_CONFIG.get(threat, base_cfg)
    cfg = {
        'rolling': int(cfg_src.get('rolling', base_cfg['rolling'])),
        'lags': [int(l) for l in cfg_src.get('lags', base_cfg['lags']) if int(l) in (1, 2, 6, 24)]
    }
    if not cfg['lags']:
        cfg['lags'] = [1, 2, 6]
    for lag in cfg['lags']:
        subset[f'lag{lag}'] = subset['y_log'].shift(lag)
    subset['rolling_mean'] = subset['y_log'].rolling(cfg['rolling']).mean().shift(1)
    subset['rolling_std'] = subset['y_log'].rolling(cfg['rolling']).std().shift(1)
    if 'rolling_sum24h' in subset.columns:
        subset = subset.drop(columns=['rolling_sum24h'])
    return subset, cfg

def _enough_history(subset, horizon_hours: int) -> bool:
    return len(subset) >= max(200, int(horizon_hours * 3))

@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    return joblib.load(path)

def train_xgb_for_threat(master_df: pd.DataFrame, threat: str, test_days: int = 7):
    grouped = build_hourly_counts(master_df)
    if grouped.empty:
        return None
    sub = grouped[grouped['Threat Type'] == threat].copy()
    if len(sub) < 150:
        return None
    thr = sub['y'].quantile(0.98)
    sub = sub[sub['y'] <= thr]
    sub['ds'] = pd.to_datetime(sub['ds'], errors='coerce')
    sub = _add_time_features(sub)
    sub = _merge_extra_columns(sub, master_df, threat)
    sub, cfg = _add_lags_rolls(sub, threat)

    feature_cols = [
        'hour', 'dayofweek', 'is_weekend', 'is_night', 'weekofyear',
        'time_since_last', 'Severity', 'attack_result_label', 'direction', 'duration'
    ]
    feature_cols += [c for c in ['lag1', 'lag2', 'lag6', 'lag24'] if c in sub.columns]
    for c in ['rolling_mean', 'rolling_std', 'rolling_sum24h']:
        if c in sub.columns:
            feature_cols.append(c)

    sub = sub.dropna(subset=feature_cols)
    if sub.empty:
        return None

    cutoff = sub['ds'].max() - pd.Timedelta(days=test_days)
    train = sub[sub['ds'] <= cutoff]
    test = sub[sub['ds'] > cutoff]
    if len(train) < 50 or len(test) < 20:
        return None

    X_full = train[feature_cols]
    y_full = train['y_log']
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=1000, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        objective='reg:squarederror', n_jobs=4
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_pred_log = model.predict(X_val)
    val_mae = float(np.mean(np.abs(np.expm1(val_pred_log) - np.expm1(y_val))))
    val_rmse = float(np.sqrt(np.mean((np.expm1(val_pred_log) - np.expm1(y_val))**2)))
    resid_std_log = float(np.std(y_val - val_pred_log))

    model_path = os.path.join(MODELS_DIR, f"xgb_{re.sub('[^A-Za-z0-9]+','_', threat)}.joblib")
    joblib.dump(
        {"model": model, "features": feature_cols, "cfg": cfg, "resid_std_log": resid_std_log},
        model_path
    )
    return {
        "model_path": model_path, "features": feature_cols, "cfg": cfg, "resid_std_log": resid_std_log,
        "train": train, "test": test,
        "validation": {"val_mae": val_mae, "val_rmse": val_rmse, "resid_std_log": resid_std_log}
    }

def forecast_recursive(
    master_df: pd.DataFrame,
    threat: str,
    horizon_days: int,
    model_bundle: dict,
    seasonality_strength: Optional[float] = None,
    noise_level: Optional[float] = None,
    spike_prob: Optional[float] = None,
    seed: int = 1234,
):
    rng = np.random.default_rng(seed)
    grouped = build_hourly_counts(master_df)
    sub = grouped[grouped['Threat Type'] == threat].copy()
    sub['ds'] = pd.to_datetime(sub['ds'], errors="coerce")
    sub = _add_time_features(sub)
    sub = _merge_extra_columns(sub, master_df, threat)
    sub, _ = _add_lags_rolls(sub, threat)

    feature_cols = model_bundle["features"]
    model = model_bundle["model"]
    resid_std_log = float(model_bundle.get("resid_std_log", 0.10))

    history = sub.dropna(subset=feature_cols).copy().sort_values("ds")
    if history.empty:
        return None

    horizon_hours = int(horizon_days * 24)
    if not _enough_history(history, horizon_hours):
        return {"insufficient_history": True, "needed": max(200, horizon_hours * 10), "available": len(history)}

    if seasonality_strength is None or noise_level is None or spike_prob is None:
        tail = history.tail(min(len(history), 24*14))
        by_hour = tail.groupby(tail['ds'].dt.hour)['y_log'].mean()
        amp = float(by_hour.max() - by_hour.min()) if not by_hour.empty else 0.0
        seasonality_strength = float(np.clip(amp / 1.2, 0.30, 0.85)) if seasonality_strength is None else seasonality_strength
        noise_level = float(np.clip(resid_std_log / 0.35, 0.15, 0.60)) if noise_level is None else noise_level
        y_tail = np.expm1(tail['y_log'])
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
            "hour": hour, "dayofweek": dayofweek, "is_weekend": is_weekend,
            "is_night": is_night, "weekofyear": weekofyear,
            "time_since_last": 1.0,
        }
        for c in ["Severity", "attack_result_label", "direction", "duration"]:
            row[c] = history[c].iloc[-1] if c in history.columns else 0
        for c in feature_cols:
            if c.startswith("lag"):
                k = int(c[3:])
                row[c] = ylog_series[-k] if len(ylog_series) >= k else np.nan
        roll_n = model_bundle["cfg"].get("rolling", 3)
        if len(ylog_series) >= roll_n:
            row["rolling_mean"] = float(pd.Series(ylog_series[-roll_n:]).mean())
            row["rolling_std"] = float(pd.Series(ylog_series[-roll_n:]).std(ddof=0))
        else:
            row["rolling_mean"] = np.nan
            row["rolling_std"] = np.nan
        if len(ylog_series) >= 24:
            row["rolling_sum24h"] = float(pd.Series(ylog_series[-24:]).sum())
        else:
            row["rolling_sum24h"] = np.nan

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

def plot_recent_and_forecast(
    threat: str,
    recent_actual: pd.DataFrame,
    fcst_df: pd.DataFrame,
    lookback_hours: int = 48,
    title_suffix: str = ""
):
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

def pretrain_models(master_df: pd.DataFrame):
    threats = sorted(master_df['Threat Type'].dropna().astype(str).unique())
    if not threats:
        with st.sidebar:
            st.info("No threat types found to pretrain.")
        return
    with st.sidebar:
        st.markdown("### Pretraining models")
        prog = st.progress(0.0, text="Starting…")
    total = len(threats)
    for i, threat in enumerate(threats, start=1):
        with st.spinner(f"Training {threat} ({i}/{total})…"):
            bundle = train_xgb_for_threat(master_df, threat, test_days=7)
        with st.sidebar:
            if bundle is None:
                st.warning(f"⏭️ Not enough data for {threat}")
            else:
                st.write(f"✅ {threat} trained")
        prog.progress(i/total, text=f"{i}/{total} models ready")
    with st.sidebar:
        st.success("Pretraining complete.")


# =============== UI ===============
st.title("🛡️ Predicción de Ataques")
st.caption("Subir CSV, ZIP (con CSV) o .gz → Procesar → Entrenar → Predecir")

# One-time seed bootstrap
if "seed_done" not in st.session_state:
    with st.status("Initializing data (first run only)…", expanded=True) as s:
        try:
            _bootstrap_seed_data()
            st.session_state["seed_done"] = True
            s.update(label="Initialization complete", state="complete")
        except Exception as e:
            s.update(label="Initialization failed", state="error")
            st.error(f"Bootstrap failed: {e}")

# Sidebar: Data status
with st.sidebar:
    st.header("📦 Data Status")
    master = read_master_cached()
    cov = _coverage_stats(master)
    if cov:
        start, end, n = cov
        st.success(f"Data from **{start}** to **{end}**  \nRows: **{n:,}**")
        tt_list = sorted(list(map(str, master.get("Threat Type", pd.Series(dtype=str)).dropna().unique())))
        st.write(f"Threat Types ({len(tt_list)}):")
        st.write(", ".join(tt_list[:30]) + (" ..." if len(tt_list) > 30 else ""))

        try:
            snapshot_path = os.path.join(DATA_DIR, "master_snapshot.parquet")
            pq.write_table(pa.Table.from_pandas(master, preserve_index=False), snapshot_path, compression="zstd")
            st.download_button("⬇️ Download master.parquet",
                               data=open(snapshot_path, "rb").read(),
                               file_name="master.parquet",
                               mime="application/octet-stream")
        except Exception as e:
            st.warning(f"Could not create Parquet snapshot: {e}")

        if os.path.exists(MASTER_CSV):
            try:
                st.download_button(
                    "⬇️ Download legacy master.csv",
                    data=open(MASTER_CSV, "rb").read(),
                    file_name="master.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.warning(f"Could not read legacy master.csv: {e}")
    else:
        st.info("No data yet. Upload a CSV, ZIP (with CSV), or compressed .gz file to get started.")

    if st.button("↻ Clear caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caches cleared. Reloading…")
        st.rerun()

# 1) Upload & Processing
st.subheader("1) Agrega Información para Entrenar el Modelo")
st.markdown("""
Sube un archivo CSV, ZIP (conteniendo un CSV), o comprimido (.gz). Archivos grandes (>200MB descomprimidos) se procesan automáticamente en partes.
- **Recomendación**: Comprime tu CSV a .gz o ZIP para archivos grandes o divide en partes si supera 200MB descomprimido.
- **Nota**: En Streamlit Community Cloud, el límite es 200MB para el archivo subido.
""")

fast_mode = st.toggle("🚀 Carga rápida (omite resumen de recurrencia)", value=True)
chunksize_opt = st.select_slider("Tamaño de chunk para procesar", options=[25_000, 50_000, 100_000, 150_000, 200_000], value=50_000, format_func=lambda x: f"{x:,} filas")

uploaded_file = st.file_uploader("Subir CSV, ZIP o .gz", type=["csv", "zip", "gz"])

colA, colB = st.columns([1, 1])
with colA:
    default_outname = dt.datetime.now().strftime("processed_%Y%m%d_%H%M%S.csv")
    outname = st.text_input("Nombre del archivo procesado", value=default_outname)
with colB:
    process_btn = st.button("Procesar y Fusionar", type="primary", use_container_width=True, disabled=not uploaded_file)

if process_btn and uploaded_file:
    master = read_master_cached()
    before_rows = len(master)
    total_rows_processed = 0
    processed_files = []

    with st.status("Procesando archivo…", expanded=True) as status:
        raw_path = os.path.join(DATA_DIR, f"upload_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")

        if uploaded_file.name.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
                for csv_name in [name for name in zip_ref.namelist() if name.endswith(".csv")]:
                    raw_path_part = raw_path + f"_part_{os.path.basename(csv_name).replace('.csv', '')}.csv"
                    with zip_ref.open(csv_name) as f_in, open(raw_path_part, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
                    processed_path = os.path.join(PROCESSED_DIR, f"{outname}_{os.path.basename(csv_name)}")
                    result = process_log_csv_with_progress(raw_path_part, processed_path, chunksize=chunksize_opt, fast_mode=fast_mode)
                    master = _update_master_with_processed(result)
                    total_rows_processed += result["rows"]
                    processed_files.append(processed_path)
                    st.write(f"Archivo {csv_name}: {result['rows']:,} filas procesadas.")

        elif uploaded_file.name.endswith(".gz"):
            raw_path += ".csv"
            with gzip.open(uploaded_file, "rb") as f_in, open(raw_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
            file_size_mb = os.path.getsize(raw_path) / (1024 * 1024)
            if file_size_mb > 200:
                st.warning(f"Archivo descomprimido grande ({file_size_mb:.2f}MB). Procesando en partes.")
            processed_path = os.path.join(PROCESSED_DIR, outname)
            result = process_log_csv_with_progress(raw_path, processed_path, chunksize=chunksize_opt, fast_mode=fast_mode)
            master = _update_master_with_processed(result)
            total_rows_processed = result["rows"]
            processed_files.append(processed_path)
            st.write(f"Archivo {uploaded_file.name}: {result['rows']:,} filas procesadas.")

        else:  # CSV
            raw_path += ".csv"
            with open(raw_path, "wb") as f_out:
                shutil.copyfileobj(uploaded_file, f_out, length=16 * 1024 * 1024)
            file_size_mb = os.path.getsize(raw_path) / (1024 * 1024)
            if file_size_mb > 200:
                st.warning(f"Archivo descomprimido grande ({file_size_mb:.2f}MB). Procesando en partes.")
            processed_path = os.path.join(PROCESSED_DIR, outname)
            result = process_log_csv_with_progress(raw_path, processed_path, chunksize=chunksize_opt, fast_mode=fast_mode)
            master = _update_master_with_processed(result)
            total_rows_processed = result["rows"]
            processed_files.append(processed_path)
            st.write(f"Archivo {uploaded_file.name}: {result['rows']:,} filas procesadas.")

        status.update(label=f"Procesamiento completado ✅ ({total_rows_processed:,} filas)", state="complete")

    after_rows = len(master)
    st.metric("Filas en master", value=f"{after_rows:,}", delta=f"+{after_rows - before_rows:,}")
    for processed_path in processed_files:
        st.download_button(
            f"⬇️ Descargar {os.path.basename(processed_path)}",
            data=open(processed_path, "rb").read(),
            file_name=os.path.basename(processed_path),
            mime="text/csv"
        )
    st.session_state["_refresh_after_merge"] = True
    st.rerun()


# 1.5) Optional pretraining
master = read_master_cached()
if not master.empty:
    c1, _ = st.columns([1, 3])
    with c1:
        if st.button("⚙️ Entrenar previamente todos los modelos (opcional)", use_container_width=True):
            pretrain_models(master)

# 2) Train & Forecast
st.subheader("2) Entrenar el modelo y generar predicciones")
st.write("Seleccione **Tipo(s) de amenaza**, elija **horizonte** (7/14/30 días) y cree gráficos de validación y pronóstico.")

if master.empty:
    st.warning("Cargue y procese al menos un CSV primero.")
else:
    threats = sorted(list(map(str, master['Threat Type'].dropna().astype(str).unique())))
    chosen = st.multiselect("Elija los tipos de amenazas para entrenar", options=threats, default=threats[:1])
    horizon_choice = st.select_slider("Horizonte de previsión", options=[7, 14, 30], value=7, format_func=lambda d: f"{d} days")
    lookback_hours = st.select_slider(
        "Historial real para mostrar antes del pronóstico",
        options=[24, 48, 72, 96, 120, 144, 168],
        value=48,
        format_func=lambda h: f"{h//24} días"
    )
    run_btn = st.button("Entrene y Pronostico", type="primary", use_container_width=True, disabled=len(chosen)==0)

    if run_btn:
        grouped_all = hourly_counts_cached(master)
        for threat in chosen:
            with st.spinner(f"Entrenando {threat}…"):
                bundle = train_xgb_for_threat(master, threat, test_days=7)
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
                    f"**{threat}**: No hay suficiente historial con funciones listas para {horizon_choice}d "
                    f"(needed ~{fcst['needed']}, available {fcst['available']}). Prueba un horizonte más corto."
                )
                continue

            grouped = grouped_all[grouped_all["Threat Type"] == threat].copy()
            grouped["ds"] = pd.to_datetime(grouped["ds"], errors="coerce")
            fcst_start = (
                fcst["ds"].min()
                if (fcst is not None and not fcst.empty)
                else grouped["ds"].max() + pd.Timedelta(hours=1)
            )
            recent_actual = grouped[
                (grouped["ds"] >= fcst_start - pd.Timedelta(hours=lookback_hours)) &
                (grouped["ds"] < fcst_start)
            ]

            fig = plot_recent_and_forecast(
                threat,
                recent_actual,
                fcst,
                lookback_hours=lookback_hours,
                title_suffix=f"(+{horizon_choice}d)"
            )
            st.pyplot(fig)

            plot_path = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_choice}d.png")
            fig.savefig(plot_path, dpi=160)
            if fcst is not None and not isinstance(fcst, dict) and not fcst.empty:
                csv_path = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_choice}d_fcst.csv")
                fcst.to_csv(csv_path, index=False)
            else:
                csv_path = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_choice}d_fcst.csv")
                pd.DataFrame(columns=["ds", "y_hat"]).to_csv(csv_path, index=False)

            val = bundle["validation"]
            st.caption(f"Validation (internal, last split): MAE={val['val_mae']:.2f}  |  RMSE={val['val_rmse']:.2f}")
            cdl, cdp = st.columns(2)
            with cdl:
                st.download_button("⬇️ Download forecast CSV", data=open(csv_path, "rb").read(),
                                   file_name=os.path.basename(csv_path), mime="text/csv")
            with cdp:
                st.download_button("⬇️ Download plot PNG", data=open(plot_path, "rb").read(),
                                   file_name=os.path.basename(plot_path), mime="image/png")

# Notes
st.divider()
st.subheader("Notas y Limitaciones")
st.markdown("""
- **Archivos grandes**: Archivos descomprimidos >200MB se procesan en partes automáticamente.
- **Persistencia**: Los datos procesados se añaden a `data/master_parquet/` como partes Parquet.
- **Consejo**: Ajusta el tamaño de chunk si encuentras problemas de memoria.
- **Despliegue**: Para archivos muy grandes, considera despliegue local y elevar `server.maxUploadSize`.
""")
st.caption("© Streamlit + XGBoost")
