# app.py — Cyber Attack Forecasting Tool (Streamlit + XGBoost)

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
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds   # (imported for future use)
import pyarrow.compute as pc
import pyarrow.types as patypes  # (imported for future use)

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.autolayout": True})

# =========================
# ---- CONFIG / STORAGE ----
# =========================
st.set_page_config(page_title="Cyber Attacks Forecaster", page_icon="🛡️", layout="wide")

DATA_DIR = "data"
MODELS_DIR = "models"
PLOTS_DIR = "plots"
PROCESSED_DIR = "processed"
SEEDS_DIR = "seeds"  # put baseline CSVs here if you want auto-preload
MASTER_DS_DIR = os.path.join(DATA_DIR, "master_parquet")   # folder with many parquet parts
ENRICH_SUFFIX_PARQUET = "_enriched_raw.parquet"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SEEDS_DIR, exist_ok=True)
os.makedirs(MASTER_DS_DIR, exist_ok=True)

MASTER_CSV = os.path.join(DATA_DIR, "master.csv")
SEED_FLAG = os.path.join(DATA_DIR, ".seeded")

# ---- Optional: seed dataset from URL in Secrets on first run ----
import requests
DATA_URL = st.secrets.get("DATA_URL", "")
SKIP_BOOTSTRAP = st.secrets.get("SKIP_BOOTSTRAP", "0") == "1"


def _have_any_master() -> bool:
    """Do we already have any master data?"""
    return bool(glob.glob(os.path.join(MASTER_DS_DIR, "*.parquet"))) or os.path.exists(MASTER_CSV)


def ensure_secret_seed_download():
    """If DATA_URL is set and no master/seed exists, download one CSV into seeds/."""
    if not DATA_URL or _have_any_master() or glob.glob(os.path.join(SEEDS_DIR, "*.csv")):
        return
    os.makedirs(SEEDS_DIR, exist_ok=True)
    dest = os.path.join(SEEDS_DIR, "seed_from_link.csv")
    if os.path.exists(dest):
        return
    with st.spinner("Downloading initial dataset (first run only)…"):
        with requests.get(DATA_URL, stream=True, timeout=300) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            done, chunk = 0, 8 * 1024 * 1024  # 8 MB
            prog = st.progress(0)
            with open(dest, "wb") as f:
                for part in r.iter_content(chunk_size=chunk):
                    if part:
                        f.write(part)
                        done += len(part)
                        if total:
                            prog.progress(min(done / total, 1.0))
            prog.empty()
    st.success("Seed CSV downloaded. It will be merged into master on this run.")


def _bust_caches_and_rerun():
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()


# =================================
# ---- 1) PROCESSING FUNCTIONS ----
# =================================

def _add_enriched_parquet_to_master(parquet_path: str):
    """Copy the parquet file into the master dataset folder as a new part."""
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(MASTER_DS_DIR, f"part-{ts}.parquet")
    shutil.copyfile(parquet_path, dst)
    return dst


@st.cache_data(show_spinner=False)
def read_master_cached():
    return _read_master()


def _read_master_parquet() -> pd.DataFrame:
    """Robustly load all parquet parts under MASTER_DS_DIR with unified schema."""
    part_paths = sorted(glob.glob(os.path.join(MASTER_DS_DIR, "*.parquet")))
    if not part_paths:
        return pd.DataFrame()

    qdir = os.path.join(MASTER_DS_DIR, "_quarantine")
    os.makedirs(qdir, exist_ok=True)

    def _norm_type(t: pa.DataType) -> pa.DataType:
        if pa.types.is_dictionary(t):
            return t.value_type
        return t

    def _type_tag(t: pa.DataType) -> str:
        t = _norm_type(t)
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            return "str"
        if pa.types.is_timestamp(t):
            return "ts"
        if pa.types.is_floating(t):
            return "float"
        if pa.types.is_integer(t):
            return "int"
        if pa.types.is_boolean(t):
            return "bool"
        return "other"

    observed: dict[str, set[str]] = {}
    samples: dict[str, pa.DataType] = {}
    good_paths = []

    for p in part_paths:
        try:
            t = pq.read_table(p)
            for f in t.schema:
                tag = _type_tag(f.type)
                observed.setdefault(f.name, set()).add(tag)
                samples.setdefault(f.name, _norm_type(f.type))
            good_paths.append(p)
        except Exception as e:
            try:
                shutil.move(p, os.path.join(qdir, os.path.basename(p)))
            except Exception:
                pass
            st.warning(f"Quarantined bad parquet part: {os.path.basename(p)} — {e}")

    if not good_paths:
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
        elif "ts" in tags and ("int" in tags or "float" in tags or "other" in tags):
            targets[name] = pa.large_string()
        elif tags == {"bool"}:
            targets[name] = pa.bool_()
        else:
            targets[name] = pa.large_string()

    mixed = {k: v for k, v in observed.items() if len(v) > 1}
    if mixed:
        st.caption(
            "Note: unified mixed column types → "
            + ", ".join([f"{k}:{'/'.join(sorted(v))}" for k, v in mixed.items()])
        )

    tables = []
    for p in good_paths:
        try:
            t = pq.read_table(p)

            arrays = []
            names = []
            num_rows = t.num_rows

            for name, target in targets.items():
                if name in t.column_names:
                    col = t[name]
                    if hasattr(col, "chunks"):
                        casted_chunks = [pc.cast(ch, target) for ch in col.chunks]
                        arr = pa.chunked_array(casted_chunks, type=target)
                    else:
                        arr = pc.cast(col, target)
                else:
                    arr = pa.nulls(num_rows, type=target)
                arrays.append(arr)
                names.append(name)

            t2 = pa.table(arrays, names=names)
            tables.append(t2)

        except Exception as e:
            try:
                shutil.move(p, os.path.join(qdir, os.path.basename(p)))
            except Exception:
                pass
            st.warning(f"Quarantined bad parquet part during cast: {os.path.basename(p)} — {e}")

    if not tables:
        return pd.DataFrame()

    table = pa.concat_tables(tables, promote=True)
    df = table.to_pandas()

    if "Attack Start Time" in df.columns:
        df["Attack Start Time"] = pd.to_datetime(df["Attack Start Time"], errors="coerce")
        df["Day"] = df["Attack Start Time"].dt.date
        df["Hour"] = df["Attack Start Time"].dt.hour

    df = _dedupe_events_by_signature_time(df)
    return df


def coalesce_columns(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    base = base.copy()
    for c in extra.columns:
        if c in base.columns:
            base[c] = base[c].where(base[c].notna(), extra[c])
        else:
            base[c] = extra[c]
    return base


def ensure_unique(base: pd.DataFrame) -> pd.DataFrame:
    base = base.reset_index(drop=True)
    return base.loc[:, ~base.columns.duplicated()]


def get_first_series(df: pd.DataFrame, colname: str):
    if colname not in df.columns:
        return None
    obj = df[colname]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0]
    return obj


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


def parse_addition_info_column(df: pd.DataFrame) -> pd.DataFrame:
    def parse(info_str):
        if pd.isna(info_str):
            return {}
        pattern = r'type=([\w\[\]\."]+)\s+value=([\w\.\[\]":\-@/\\]+)'
        matches = re.findall(pattern, info_str)
        return {key.strip(): value.strip() for key, value in matches}

    parsed_dicts = df["Addition Info"].apply(parse)
    parsed_df = pd.json_normalize(parsed_dicts)
    return coalesce_columns(df, parsed_df)


def map_attack_result(df: pd.DataFrame) -> pd.DataFrame:
    result_map = {"1": "Attempted", "2": "Successful", 1: "Attempted", 2: "Successful"}
    s = get_first_series(df, "attack_result")
    if s is None:
        df["attack_result_label"] = np.nan
        return df
    df["attack_result_label"] = s.map(result_map)
    return df


@st.cache_resource(show_spinner=False)
def load_model_cached(path: str):
    return joblib.load(path)


def create_attack_signature(df: pd.DataFrame) -> pd.DataFrame:
    signature_cols = [
        "Threat Name",
        "Threat Type",
        "Threat Subtype",
        "Severity",
        "Source IP",
        "Destination IP",
        "Attacker",
        "Victim",
    ]
    for c in signature_cols:
        if c not in df.columns:
            df[c] = np.nan
    df["attack_signature"] = df[signature_cols].astype(str).agg("|".join, axis=1)
    return df


def calculate_recurrence(df: pd.DataFrame) -> pd.DataFrame:
    df["Attack Start Time"] = pd.to_datetime(df["Attack Start Time"], errors="coerce")
    grouped = df.groupby("attack_signature")
    first_rows = grouped.first().reset_index()

    agg_info = grouped["Attack Start Time"].agg(["min", "max", "count"]).reset_index()
    agg_info.columns = ["attack_signature", "First Seen", "Last Seen", "Recurrence Index"]
    agg_info["Time Frame (hrs)"] = (
        agg_info["Last Seen"] - agg_info["First Seen"]
    ).dt.total_seconds() / 3600

    def _avg_gap(row):
        if row["Recurrence Index"] > 1 and row["Time Frame (hrs)"] > 0.01:
            return row["Time Frame (hrs)"] / (row["Recurrence Index"] - 1)
        return None

    agg_info["Avg Time Between Events (hrs)"] = agg_info.apply(_avg_gap, axis=1)
    return pd.merge(first_rows, agg_info, on="attack_signature")

def _quick_row_count(parquet_dir: str) -> int:
    """Sum row counts from parquet footers (no heavy read)."""
    total = 0
    for p in glob.glob(os.path.join(parquet_dir, "*.parquet")):
        try:
            md = pq.read_metadata(p)
            total += md.num_rows or 0
        except Exception:
            pass
    return total


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

    df = parse_addition_info_column(df)
    df = map_attack_result(df)
    df = create_attack_signature(df)
    df["Attack Start Time"] = pd.to_datetime(df["Attack Start Time"], errors="coerce")
    df["Day"] = df["Attack Start Time"].dt.date
    df["Hour"] = df["Attack Start Time"].dt.hour

    raw_out = output_path.replace(".csv", "_enriched_raw.csv")
    df.to_csv(raw_out, index=False)

    df_final = calculate_recurrence(df)
    df_final = df_final.merge(df[["attack_signature", "Day", "Hour"]], on="attack_signature", how="left")
    df_final.to_csv(output_path, index=False)

    return df


def process_log_csv_with_progress(input_path: str, output_path: str, chunksize: int = 100_000, fast_mode: bool = True):
    """Stream-read CSV in chunks → enrich → append to one parquet; write small summary."""
    prog = st.progress(0.0, text="Leyendo CSV…")
    rows_done = 0

    out_parquet = output_path.replace(".csv", ENRICH_SUFFIX_PARQUET)
    Path(out_parquet).unlink(missing_ok=True)

    addinfo_re = re.compile(r'type=(?P<key>[^ \t]+)\s+value=(?P<val>[^;,\n]+)')

    def _vectorized_parse(df_chunk: pd.DataFrame) -> pd.DataFrame:
        s = df_chunk["Addition Info"].fillna("")
        ext = (
            s.str.extractall(addinfo_re)
            .reset_index()
            .rename(columns={"level_0": "row", "key": "k", "val": "v"})
        )
        if ext.empty:
            return df_chunk
        wide = ext.pivot(index="row", columns="k", values="v")
        wide.columns = [str(c).strip() for c in wide.columns]
        wide = wide.reset_index()
        out = (
            df_chunk.reset_index(drop=True)
            .reset_index()
            .merge(wide, left_on="index", right_on="row", how="left")
            .drop(columns=["index", "row"])
        )
        return out

    writer = None
    schema = None
    cols_ref = None

  # --- stream the CSV in chunks ---
    for i, df in enumerate(pd.read_csv(input_path, low_memory=False, chunksize=chunksize)):
        try:
            # 1) NEW: normalize headers on every chunk (prevents Arrow duplicate-name errors)
            df = _normalize_and_uniquify_columns(df)
    
            # 2) your existing guards for required columns
            if "Addition Info" not in df.columns:
                df["Addition Info"] = np.nan
            if "attack_result" not in df.columns:
                df["attack_result"] = np.nan
            if "Attack Start Time" not in df.columns:
                if "First Seen" in df.columns:
                    df["Attack Start Time"] = pd.to_datetime(df["First Seen"], errors="coerce")
                else:
                    raise ValueError("CSV must include 'Attack Start Time' column.")
    
            # 3) your enrichment steps
            df = _vectorized_parse(df)
            df = map_attack_result(df)
            df = create_attack_signature(df)
    
            ts = pd.to_datetime(df["Attack Start Time"], errors="coerce")
            df["Attack Start Time"] = ts
            df["Day"] = ts.dt.date
            df["Hour"] = ts.dt.hour
    
            # 4) stabilize text-ish columns for Arrow
            TEXTY_COLS = [
                "Addition Info", "Threat Name", "Threat Type", "Threat Subtype",
                "Source IP", "Destination IP", "Attacker", "Victim",
                "direction", "Severity", "attack_result", "attack_result_label"
            ]
            for c in TEXTY_COLS:
                if c in df.columns:
                    df[c] = df[c].astype("string")
    
            # 5) keep schema stable across chunks
            if cols_ref is None:
                cols_ref = list(df.columns)
            else:
                for c in cols_ref:
                    if c not in df.columns:
                        df[c] = pd.NA
                df = df.reindex(columns=cols_ref)
    
            # 6) write parquet (with first-chunk schema)
            table = pa.Table.from_pandas(df, preserve_index=False)
    
            if writer is None:
                schema = table.schema
                writer = pq.ParquetWriter(out_parquet, schema=schema, compression="zstd", use_dictionary=True)
            else:
                table = table.cast(schema)
    
            writer.write_table(table)
    
            rows_done += len(df)
            prog.progress(min(0.99, 0.02 + i * 0.02), text=f"Procesadas ~{rows_done:,} filas")
    
        except Exception as e:
            # Show chunk index + traceback so you can fix fast
            st.error(f"❌ Error while processing chunk {i:,}: {e}")
            st.exception(e)
            raise

    if fast_mode:
        pd.DataFrame({"note": ["Fast mode: resumen omitido."]}).to_csv(output_path, index=False)
    else:
        df_all = pq.read_table(out_parquet).to_pandas()
        df_final = calculate_recurrence(df_all)
        df_final = df_final.merge(df_all[["attack_signature", "Day", "Hour"]], on="attack_signature", how="left")
        df_final.to_csv(output_path, index=False)

    prog.progress(1.0, text=f"¡Listo! Total procesado: {rows_done:,} filas")
    return {"parquet_path": out_parquet, "rows": rows_done, "fast_mode": fast_mode}

# ---------- URL ingest helpers ----------
def _normalize_direct_download(url: str) -> str:
    url = url.strip()

    # Dropbox: turn ?dl=0 into a direct download
    if "dropbox.com" in url:
        if "dl=0" in url:
            url = url.replace("dl=0", "dl=1")
        elif "dl=1" not in url and "raw=1" not in url:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}dl=1"

    # Google Drive share → uc?export=download
    m = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
    if m:
        file_id = m.group(1)
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
    m = re.search(r"drive\.google\.com/open\?id=([^&]+)", url)
    if m:
        file_id = m.group(1)
        url = f"https://drive.google.com/uc?export=download&id={file_id}"

    return url


def download_url_to_csv(url: str, base_path_no_ext: str) -> str:
    """
    Streams a remote file to disk and returns a *CSV path*.
    Accepts .csv, .gz, .zip (first CSV inside), and handles Google Drive confirm pages.
    Robust to missing filename/extension.
    """
    import io, gzip, zipfile, requests

    def _looks_html(bytes_head: bytes) -> bool:
        head = bytes_head.strip().lower()
        return head.startswith(b'<!doctype html') or head.startswith(b'<html')

    def _magic(bytes_head: bytes) -> str:
        # return 'zip' | 'gz' | 'csv' | 'html' | 'unknown'
        if bytes_head[:2] == b'\x1f\x8b':
            return 'gz'
        if bytes_head[:4] == b'PK\x03\x04':
            return 'zip'
        if _looks_html(bytes_head):
            return 'html'
        # If it contains commas and newlines early, we’ll assume csv
        if b',' in bytes_head and b'\n' in bytes_head:
            return 'csv'
        return 'unknown'

    def _normalize_direct_download(url: str) -> str:
        url = url.strip()
        # Dropbox
        if "dropbox.com" in url:
            if "dl=0" in url:
                url = url.replace("dl=0", "dl=1")
            elif "dl=1" not in url and "raw=1" not in url:
                sep = "&" if "?" in url else "?"
                url = f"{url}{sep}dl=1"
        # Google Drive
        m = re.search(r"drive\.google\.com/file/d/([^/]+)", url)
        if m:
            file_id = m.group(1)
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
        m = re.search(r"drive\.google\.com/open\?id=([^&]+)", url)
        if m:
            file_id = m.group(1)
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
        return url

    def _maybe_handle_gdrive_confirm(session: requests.Session, url: str) -> requests.Response:
        """
        First request may return an HTML interstitial with a confirm token.
        If detected, extract token and re-request the file.
        """
        r = session.get(url, stream=True, timeout=600)
        r.raise_for_status()
        # If content-type is HTML or body looks like HTML, try confirm token
        ct = (r.headers.get("content-type") or "").lower()
        if "text/html" in ct:
            txt = r.text
            m = re.search(r'confirm=([0-9A-Za-z\-_]+)', txt)
            if m:
                token = m.group(1)
                # append token
                sep = "&" if "?" in url else "?"
                url2 = f"{url}{sep}confirm={token}"
                r.close()
                r = session.get(url2, stream=True, timeout=600)
                r.raise_for_status()
        return r

    url = _normalize_direct_download(url)
    raw_path = f"{base_path_no_ext}__raw.bin"
    csv_path = f"{base_path_no_ext}.csv"

    with st.status("Fetching from URL…", expanded=True) as s:
        with requests.Session() as sess:
            # Try to handle Google Drive confirm pages
            r = _maybe_handle_gdrive_confirm(sess, url)

            # show progress
            total = int(r.headers.get("content-length", 0))
            done = 0
            prog = st.progress(0.0)
            with open(raw_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        done += len(chunk)
                        if total:
                            prog.progress(min(done / total, 1.0))
            prog.empty()
        s.update(label="Download complete", state="complete")

    # Sniff first 4KB
    with open(raw_path, "rb") as fh:
        head = fh.read(4096)

    kind = _magic(head)

    try:
        if kind == "html":
            # Don’t attempt to parse; tell the user what happened.
            with open(raw_path, "rb") as fh:
                sample = fh.read(1024).decode(errors="ignore")
            st.error(
                "The downloaded file appears to be HTML (likely a Google Drive warning/confirm page) "
                "instead of a CSV/ZIP/GZ. Please make the file publicly downloadable or use a direct link."
            )
            st.code(sample)
            raise RuntimeError("Downloaded HTML page instead of data file.")

        elif kind == "gz":
            with gzip.open(raw_path, "rb") as src, open(csv_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

        elif kind == "zip":
            with zipfile.ZipFile(raw_path) as z:
                names = [n for n in z.namelist() if n.lower().endswith(".csv")]
                if not names:
                    raise RuntimeError("ZIP file has no CSV inside.")
                # pick the first CSV
                with z.open(names[0]) as src, open(csv_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)

        else:
            # Treat as CSV by default (covers plain CSV and many text/csv payloads).
            shutil.move(raw_path, csv_path)
            raw_path = None

        return csv_path

    finally:
        # Clean up temp file if still present
        try:
            if raw_path and os.path.exists(raw_path):
                os.remove(raw_path)
        except Exception:
            pass

# ===============================
# ---- 2) DATA LAYER / CACHE ----
# ===============================
def _read_master() -> pd.DataFrame:
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


def _dedupe_master(df: pd.DataFrame) -> pd.DataFrame:
    if "Attack Start Time" in df.columns:
        df = df.sort_values("Attack Start Time")
    return df.drop_duplicates(keep="first")


def _dedupe_events_by_signature_time(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    key = [c for c in ["Attack Start Time", "attack_signature"] if c in df.columns]
    if not key:
        return df
    return df.sort_values("Attack Start Time").drop_duplicates(subset=key, keep="first")

def _quick_row_count(parquet_dir: str) -> int:
    """Sum num_rows from parquet file footers without loading tables."""
    total = 0
    for p in glob.glob(os.path.join(parquet_dir, "*.parquet")):
        try:
            md = pq.read_metadata(p)
            total += md.num_rows or 0
        except Exception:
            # ignore unreadable parts here; _read_master_parquet will quarantine later
            pass
    return total

def _update_master_with_processed(enriched_info):
    """Copy new parquet into master, try to load; quarantine on failure."""
    parquet_path = enriched_info["parquet_path"] if isinstance(enriched_info, dict) else enriched_info
    new_part = _add_enriched_parquet_to_master(parquet_path)

    try:
        return _read_master_parquet()
    except MemoryError:
        st.error("Master too large to fully load right after merge (MemoryError).")
        st.info(f"Approx rows (footer-based): {_quick_row_count(MASTER_DS_DIR):,}")
        return pd.DataFrame()
    except Exception as e:
        qdir = os.path.join(MASTER_DS_DIR, "_quarantine")
        os.makedirs(qdir, exist_ok=True)
        try:
            shutil.move(new_part, os.path.join(qdir, os.path.basename(new_part)))
        except Exception:
            pass
        st.error(f"Merging failed; quarantined **{os.path.basename(new_part)}**.\n\n{e}")
        try:
            return _read_master_parquet()
        except Exception:
            return pd.DataFrame()


def _coverage_stats(df: pd.DataFrame):
    if df.empty or "Attack Start Time" not in df.columns:
        return None
    ts = df["Attack Start Time"].dropna()
    if ts.empty:
        return None
    return ts.min(), ts.max(), len(df)

# ------- Stateless (no-master) helpers -------
def _append_session_master(new_df: pd.DataFrame) -> pd.DataFrame:
    """Append new_df to a session-only master and de-duplicate by time+signature when possible."""
    base = st.session_state.get("session_master_df")
    if base is None or base.empty:
        st.session_state["session_master_df"] = _dedupe_events_by_signature_time(new_df)
    else:
        merged = pd.concat([base, new_df], ignore_index=True)
        st.session_state["session_master_df"] = _dedupe_events_by_signature_time(merged)
    return st.session_state["session_master_df"]

def _session_master_df() -> pd.DataFrame:
    return st.session_state.get("session_master_df", pd.DataFrame())

def _session_master_download_paths() -> tuple[str, str]:
    """Write the current session master to parquet & csv so user can download."""
    df = _session_master_df().copy()
    parquet_path = os.path.join(DATA_DIR, "session_master.parquet")
    csv_path = os.path.join(DATA_DIR, "session_master.csv")
    if not df.empty:
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), parquet_path, compression="zstd")
        df.to_csv(csv_path, index=False)
    return parquet_path, csv_path

# -------------------------------
# Seed/Bootstrap on first run
# -------------------------------
def _discover_seed_paths() -> list:
    candidates = []
    env_paths = os.environ.get("CYBER_SEED_CSVS", "")
    if env_paths:
        candidates.extend([p.strip() for p in env_paths.split(",") if p.strip()])
    candidates.extend(glob.glob(os.path.join(SEEDS_DIR, "*.csv")))
    for p in ["BDS_BIG_2MONTHS.csv", "BDS_UNIFICADO.csv", "BDS1.csv", "/mnt/data/BDS1.csv"]:
        if os.path.exists(p):
            candidates.append(p)
    seen, uniq = set(), []
    for c in candidates:
        if c not in seen and os.path.exists(c):
            seen.add(c)
            uniq.append(c)
    return uniq


def _merge_or_process_seed(path: str):
    outp = os.path.join(PROCESSED_DIR, f"seed_{os.path.basename(path)}")
    return process_log_csv_with_progress(path, outp, chunksize=250_000, fast_mode=True)


def _bootstrap_seed_data():
    if glob.glob(os.path.join(MASTER_DS_DIR, "*.parquet")) or os.path.exists(MASTER_CSV):
        return
    paths = _discover_seed_paths()
    if not paths:
        return
    for p in paths:
        try:
            result = _merge_or_process_seed(p)
            _update_master_with_processed(result)
        except Exception as e:
            st.warning(f"Seed load failed for {p}: {e}")
    with open(SEED_FLAG, "w") as f:
        f.write(dt.datetime.now().isoformat())


# ==================================================
# ---- 3) FEATURES / MODEL TRAINING  ----
# ==================================================
WINDOW_CONFIG = {
    "DoS": {"rolling": 3, "lags": [1, 2]},
    "Scan": {"rolling": 6, "lags": [1, 2, 6]},
    "Malicious Flow": {"rolling": 12, "lags": [1, 2, 6]},
    "Vulnerability Attack": {"rolling": 6, "lags": [1, 2, 24]},
    "Attack": {"rolling": 6, "lags": [1, 2]},
    "Malfile": {"rolling": 3, "lags": [1, 2]},
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
    extra_cols = ["Severity", "attack_result_label", "direction", "duration"]
    for c in extra_cols:
        if c not in raw_df.columns:
            raw_df[c] = np.nan
    raw_df = raw_df.copy()
    raw_df["ds"] = pd.to_datetime(raw_df["Attack Start Time"], errors="coerce").dt.floor("h")
    extra_data = raw_df[raw_df["Threat Type"].astype(str) == str(threat)][["ds"] + extra_cols].drop_duplicates(subset="ds")
    merged = pd.merge(grouped, extra_data, on="ds", how="left")

    le_sev = LabelEncoder()
    le_dir = LabelEncoder()
    merged["Severity"] = le_sev.fit_transform(merged["Severity"].astype(str))
    merged["direction"] = le_dir.fit_transform(merged["direction"].astype(str))
    merged["attack_result_label"] = pd.to_numeric(merged["attack_result_label"], errors="coerce").fillna(0)
    merged["duration"] = pd.to_numeric(merged["duration"], errors="coerce").fillna(0)
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


def add_fourier_terms(df, ts_col="ds", periods=(24, 168), K=3):
    df = df.copy()
    t = (df[ts_col].view("int64") // 10**9)  # seconds since epoch (hourly grain ok)
    for P in periods:
        Psec = P * 3600.0
        for k in range(1, K + 1):
            df[f"sin_{P}_{k}"] = np.sin(2 * np.pi * k * (t / Psec))
            df[f"cos_{P}_{k}"] = np.cos(2 * np.pi * k * (t / Psec))
    return df


def fourier_row(ts, periods=(24, 168), K=3):
    t = int(pd.Timestamp(ts).value // 10**9)
    out = {}
    for P in periods:
        Psec = P * 3600.0
        for k in range(1, K + 1):
            out[f"sin_{P}_{k}"] = np.sin(2 * np.pi * k * (t / Psec))
            out[f"cos_{P}_{k}"] = np.cos(2 * np.pi * k * (t / Psec))
    return out


def seasonal_naive_from_counts(history_counts, horizon_hours, period=24):
    hist = np.asarray(history_counts)
    return np.array([hist[-period + (h - 1) % period] for h in range(1, horizon_hours + 1)])


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
    subset["rolling_std"] = subset["y_log"].rolling(cfg["rolling"]).std().shift(1)

    if "rolling_sum24h" in subset.columns:
        subset = subset.drop(columns=["rolling_sum24h"])

    return subset, cfg


def _enough_history(subset, horizon_hours: int) -> bool:
    return len(subset) >= max(200, int(horizon_hours * 3))


def train_xgb_for_threat(master_df: pd.DataFrame, threat: str, test_days: int = 7):
    grouped = build_hourly_counts(master_df)
    if grouped.empty:
        return None

    sub = grouped[grouped["Threat Type"] == threat].copy()
    if len(sub) < 150:
        return None

    thr = sub["y"].quantile(0.98)
    sub = sub[sub["y"] <= thr]
    sub["ds"] = pd.to_datetime(sub["ds"], errors="coerce")
    sub = _add_time_features(sub)
    sub = _merge_extra_columns(sub, master_df, threat)
    sub, cfg = _add_lags_rolls(sub, threat)

    feature_cols = [
        "hour",
        "dayofweek",
        "is_weekend",
        "is_night",
        "weekofyear",
        "time_since_last",
        "Severity",
        "attack_result_label",
        "direction",
        "duration",
    ]
    feature_cols += [c for c in ["lag1", "lag2", "lag6", "lag24"] if c in sub.columns]
    for c in ["rolling_mean", "rolling_std", "rolling_sum24h"]:
        if c in sub.columns:
            feature_cols.append(c)

    sub = sub.dropna(subset=feature_cols)
    if sub.empty:
        return None

    cutoff = sub["ds"].max() - pd.Timedelta(days=test_days)
    train = sub[sub["ds"] <= cutoff]
    test = sub[sub["ds"] > cutoff]

    if len(train) < 50 or len(test) < 20:
        return None

    X_full = train[feature_cols]
    y_full = train["y_log"]
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

    model = XGBRegressor(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
        n_jobs=4,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_pred_log = model.predict(X_val)
    val_mae = float(np.mean(np.abs(np.expm1(val_pred_log) - np.expm1(y_val))))
    val_rmse = float(np.sqrt(np.mean((np.expm1(val_pred_log) - np.expm1(y_val)) ** 2)))
    resid_std_log = float(np.std(y_val - val_pred_log))

    model_path = os.path.join(MODELS_DIR, f"xgb_{re.sub('[^A-Za-z0-9]+', '_', threat)}.joblib")
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
        tail = history.tail(min(len(history), 24 * 14))
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
            "hour": hour,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "weekofyear": weekofyear,
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


def pretrain_models(master_df: pd.DataFrame):
    threats = sorted(master_df["Threat Type"].dropna().astype(str).unique())
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

        prog.progress(i / total, text=f"{i}/{total} models ready")

    with st.sidebar:
        st.success("Pretraining complete.")


# ===========================
# ---- 4) STREAMLIT UI  -----
# ===========================
st.title("🛡️ Predicción de Ataques")
st.caption("Subir Información → Procesar → Entrenar → Predecir")

# First-run seeding with visible status (prevents blank screen)
if "seed_done" not in st.session_state:
    with st.status("Initializing data (first run only)…", expanded=True) as s:
        try:
            if not SKIP_BOOTSTRAP:
                ensure_secret_seed_download()
                _bootstrap_seed_data()
            st.session_state["seed_done"] = True
            s.update(label="Initialization complete", state="complete")
            _bust_caches_and_rerun()
        except Exception as e:
            s.update(label="Initialization failed", state="error")
            st.error(f"Bootstrap failed: {e}")

# Sidebar: data status (supports stateless mode)
with st.sidebar:
    st.header("📦 Data Status")

    stateless = st.toggle(
        "No master (session only)",
        value=True,
        help="Use only the current upload(s) in this session. Nothing is stored as a server-side master."
    )

    if stateless:
        master = _session_master_df()
        cov = _coverage_stats(master)
        st.caption("Session master: in-memory only")
    else:
        st.caption(f"Master parts found: {len(glob.glob(os.path.join(MASTER_DS_DIR, '*.parquet')))}")
        master = read_master_cached()
        cov = _coverage_stats(master)

    if cov:
        start, end, n = cov
        st.success(f"Data from **{start}** to **{end}**  \nRows: **{n:,}**")
        if not master.empty and "Threat Type" in master.columns:
            tt_list = sorted(map(str, master["Threat Type"].dropna().unique()))
        else:
            tt_list = []
        st.write(f"Threat Types ({len(tt_list)}):")
        st.write(", ".join(tt_list[:30]) + (" ..." if len(tt_list) > 30 else ""))

        # Downloads
        if stateless:
            p_parq, p_csv = _session_master_download_paths()
            if os.path.exists(p_parq):
                st.download_button("⬇️ Download session master.parquet", data=open(p_parq, "rb").read(),
                                   file_name="session_master.parquet", mime="application/octet-stream")
            if os.path.exists(p_csv):
                st.download_button("⬇️ Download session master.csv", data=open(p_csv, "rb").read(),
                                   file_name="session_master.csv", mime="text/csv")
        else:
            try:
                snapshot_path = os.path.join(DATA_DIR, "master_snapshot.parquet")
                pq.write_table(pa.Table.from_pandas(master, preserve_index=False), snapshot_path, compression="zstd")
                st.download_button(
                    "⬇️ Download master.parquet",
                    data=open(snapshot_path, "rb").read(),
                    file_name="master.parquet",
                    mime="application/octet-stream",
                )
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
        st.info("No data yet. Upload a CSV to get started.")

    st.markdown("---")
    if st.button("↻ Clear caches"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Caches cleared. Reloading…")
        st.experimental_rerun()

# -----------------------
# 1) Upload & Processing
# -----------------------
st.subheader("1) Agrega Información para Entrenar el Modelo")
st.write("Sube un CSV sin procesar → se **procesará** y se **fusionará** con el conjunto de datos maestro.")

uploaded = st.file_uploader("Subir CSV (exportación BDS sin procesar)", type=["csv"])

# >>> DEFINE PERFORMANCE CONTROLS *BEFORE* ANY HANDLERS USE THEM  <<<
fast_mode = st.toggle(
    "🚀 Carga rápida (omite resumen de recurrencia ahora)",
    value=True,
    help="Guarda Parquet enriquecido y fusiona; podrás generar el resumen más tarde.",
)
chunksize_opt = st.select_slider(
    "Tamaño de chunk para procesar",
    options=[100_000, 150_000, 200_000, 250_000, 300_000],
    value=250_000,
    format_func=lambda x: f"{x:,} filas",
)

# ---- URL ingestion (bypasses 200MB uploader limit) ----
st.markdown("**Or paste a link (Dropbox / Google Drive / S3 / HTTPS):**")
url_in = st.text_input("URL to a CSV (or .gz/.zip with a CSV inside)", placeholder="https://…")
fetch_btn = st.button("Fetch & Merge from URL", use_container_width=True, disabled=not url_in)

if fetch_btn and url_in:
    # use the already-defined `master` from the sidebar block
    before_rows = len(master) if isinstance(master, pd.DataFrame) else 0

    with st.status("Downloading and processing…", expanded=True) as status:
        ts_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_no_ext = os.path.join(DATA_DIR, f"remote_{ts_tag}")
        csv_local_path = download_url_to_csv(url_in, base_no_ext)
        st.write(f"Saved to: `{csv_local_path}`")
        
        # Show a small preview to catch HTML/garbage early
        st.write("Preview of downloaded file (first ~1KB):")
        with open(csv_local_path, "rb") as fh:
            preview = fh.read(1024).decode(errors="ignore")
        st.code(preview)

        processed_path = os.path.join(PROCESSED_DIR, f"processed_{ts_tag}.csv")
        st.write("Reading & enriching CSV…")
        result = process_log_csv_with_progress(
            csv_local_path,
            processed_path,
            chunksize=chunksize_opt,
            fast_mode=fast_mode,
        )
        st.write(f"Rows enriched (RAW): **{result['rows']:,}**")
        st.write("Merging into master…")

        if stateless:
            try:
                curr = pq.read_table(result["parquet_path"]).to_pandas()
            except MemoryError:
                st.error("⚠️ The enriched parquet is too large to load into memory right now.")
                st.info("Tip: lower chunksize, keep 🚀 Fast mode on, or switch off stateless so it persists to disk.")
                curr = pd.DataFrame()
            master = _append_session_master(curr)
            after_rows = len(master)
            status.update(label="Done ✅ (session only; nothing persisted)", state="complete")

        else:
            # persisted master mode (old behavior)
            master = _update_master_with_processed(result)
            after_rows = len(master) if (isinstance(master, pd.DataFrame) and not master.empty) else _quick_row_count(MASTER_DS_DIR)
            status.update(label="Done ✅", state="complete")

    st.metric("Rows in master", value=f"{after_rows:,}", delta=f"+{after_rows - before_rows:,}")
    st.download_button(
        "⬇️ Download processed CSV",
        data=open(processed_path, "rb").read(),
        file_name=os.path.basename(processed_path),
        mime="text/csv",
    )

    # extra downloads when in stateless mode
    if stateless and after_rows > 0:
        p_parq, p_csv = _session_master_download_paths()
        if os.path.exists(p_parq):
            st.download_button(
                "⬇️ Download merged (session) parquet",
                data=open(p_parq, "rb").read(),
                file_name="session_master.parquet",
                mime="application/octet-stream",
            )
        if os.path.exists(p_csv):
            st.download_button(
                "⬇️ Download merged (session) CSV",
                data=open(p_csv, "rb").read(),
                file_name="session_master.csv",
                mime="text/csv",
            )

    st.session_state["_refresh_after_merge"] = True
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()



# ---- File upload path (subject to 200MB Streamlit limit) ----
colA, colB = st.columns([1, 1])
with colA:
    default_outname = dt.datetime.now().strftime("processed_%Y%m%d_%H%M%S.csv")
    outname = st.text_input("Nombre del archivo procesado", value=default_outname)
with colB:
    process_btn = st.button("Process & Merge", type="primary", use_container_width=True, disabled=uploaded is None)

if process_btn and uploaded is not None:
    before_rows = len(master) if isinstance(master, pd.DataFrame) else 0

    with st.status("Procesando archivo…", expanded=True) as status:
        raw_path = os.path.join(DATA_DIR, f"upload_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        uploaded.seek(0)
        with open(raw_path, "wb") as dst:
            shutil.copyfileobj(uploaded, dst, length=16 * 1024 * 1024)
        st.write(f"Guardado en: `{raw_path}`")

        processed_path = os.path.join(PROCESSED_DIR, outname)
        st.write("Leyendo y enriqueciendo CSV…")
        result = process_log_csv_with_progress(
            raw_path,
            processed_path,
            chunksize=chunksize_opt,
            fast_mode=fast_mode
        )
        st.write(f"Filas enriquecidas (RAW): **{result['rows']:,}**")
        st.write("Fusionando con el conjunto maestro…")

        if stateless:
            try:
                curr = pq.read_table(result["parquet_path"]).to_pandas()
            except MemoryError:
                st.error("⚠️ The enriched parquet is too large to load into memory right now.")
                st.info("Tip: lower chunksize, keep 🚀 Fast mode on, or switch off stateless so it persists to disk.")
                curr = pd.DataFrame()
            master = _append_session_master(curr)
            after_rows = len(master)
            status.update(label="Done ✅ (session only; nothing persisted)", state="complete")
        else:
            master = _update_master_with_processed(result)
            after_rows = len(master) if (isinstance(master, pd.DataFrame) and not master.empty) else _quick_row_count(MASTER_DS_DIR)
            status.update(label="Procesamiento completado ✅", state="complete")

    st.metric("Filas en master", value=f"{after_rows:,}", delta=f"+{after_rows - before_rows:,}")
    st.download_button(
        "⬇️ Descargar CSV procesado",
        data=open(processed_path, "rb").read(),
        file_name=os.path.basename(processed_path),
        mime="text/csv",
    )

    if stateless and after_rows > 0:
        p_parq, p_csv = _session_master_download_paths()
        if os.path.exists(p_parq):
            st.download_button(
                "⬇️ Descargar combinado (sesión) parquet",
                data=open(p_parq, "rb").read(),
                file_name="session_master.parquet",
                mime="application/octet-stream",
            )
        if os.path.exists(p_csv):
            st.download_button(
                "⬇️ Descargar combinado (sesión) CSV",
                data=open(p_csv, "rb").read(),
                file_name="session_master.csv",
                mime="text/csv",
            )

    st.session_state["_refresh_after_merge"] = True
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()

st.divider()


# -----------------------
# 1.5) (Optional) Pretrain all models
# -----------------------
if not master.empty:
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("⚙️ Entrenar previamente todos los modelos (opcional)", use_container_width=True):
            pretrain_models(master)

# -----------------------
# 2) Training & Forecast
# -----------------------
st.subheader("2) Entrenar el modelo y generar predicciones")
st.write("Seleccione **Tipo(s) de amenaza**, elija **horizonte** (7/14/30 días) y cree gráficos de validación y pronóstico.")

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
                    f"**{threat}**: No hay suficiente historial con funciones listas para usar {horizon_choice}d "
                    f"(needed ~{fcst['needed']}, available {fcst['available']}). Try a shorter horizon."
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

            plot_path = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_choice}d.png")
            fig.savefig(plot_path, dpi=160)
            csv_path = os.path.join(PLOTS_DIR, f"{re.sub('[^A-Za-z0-9]+','_', threat)}_{horizon_choice}d_fcst.csv")
            fcst.to_csv(csv_path, index=False)

            val = bundle["validation"]
            st.caption(f"Validation (internal, last split): MAE={val['val_mae']:.2f}  |  RMSE={val['val_rmse']:.2f}")

            c1, c2 = st.columns(2)
            with c1:
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
st.subheader("Notes & Guardrails")
st.markdown(
    """
- **Valores atípicos**: se recorta el 2% superior de los recuentos por hora.
- **Límites de horizonte**: 7/14/30 días con comprobación de suficiencia del historial.
- **Las funciones adicionales** (`Severity`, `attack_result_label`, `direction`, `duration`) se fusionan por hora cuando están disponibles.
- **Persistencia**: los datos sembrados/fusionados se guardan en `data/master_parquet/` y se reutilizan en los reinicios.
- **Bootstrapping**: se colocan los CSV de referencia en `seeds/` o se establece `CYBER_SEED_CSVS="/path/a.csv,/path/b.csv"`.
"""
)

st.caption("© Streamlit + XGBoost")
