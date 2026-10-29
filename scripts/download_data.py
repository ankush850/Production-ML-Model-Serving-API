"""Download the UCI Online Retail II dataset into data/raw/.

Dataset: "Online Retail II" (DOI 10.24432/C5CG6D, CC BY 4.0), all transactions of a
UK online retailer between 2009-12-01 and 2011-12-09. It is a real, public,
tabular dataset with a datetime column (InvoiceDate), categorical and numeric
features, and natural missing values (CustomerID ~22%, Description ~14%).

The source ships as a 43.5 MB xlsx inside a zip; we extract both sheets, keep
the columns the pipeline needs and write one CSV into data/raw/ (gitignored).
Nothing here is synthetic.
"""

from __future__ import annotations

import io
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATASET_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "Country"]
# the II xlsx uses slightly different headers than the classic dataset
COLUMN_RENAME = {"Invoice": "InvoiceNo", "Price": "UnitPrice", "Customer ID": "CustomerID"}
RAW_DIR = Path("data/raw")
ZIP_CACHE = RAW_DIR / "online_retail_ii.zip"
OUT_CSV = RAW_DIR / "online_retail.csv"


def download_zip(url: str = DATASET_URL) -> bytes:
    """Stream the dataset zip from the UCI repository (cached under data/raw/)."""
    if ZIP_CACHE.exists():
        print(f"using cached {ZIP_CACHE}")
        return ZIP_CACHE.read_bytes()
    print(f"downloading {url} ...")
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    buf = io.BytesIO()
    total = 0
    for chunk in resp.iter_content(chunk_size=1 << 20):
        buf.write(chunk)
        total += len(chunk)
        print(f"  {total / 1e6:.1f} MB", end="\r", flush=True)
    print()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ZIP_CACHE.write_bytes(buf.getvalue())
    return buf.getvalue()


def extract_transactions(zip_bytes: bytes) -> pd.DataFrame:
    """Parse the xlsx inside the zip and return line-item transactions."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        member = zf.namelist()[0]
        with zf.open(member) as fh:
            tmp = Path(RAW_DIR) / Path(member).name
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            with tmp.open("wb") as out:
                shutil.copyfileobj(fh, out)
    sheets = pd.read_excel(tmp, sheet_name=None, engine="openpyxl")
    tmp.unlink()
    return pd.concat(sheets.values(), ignore_index=True)


def main() -> int:
    """Download, normalize and persist the raw dataset CSV."""
    if OUT_CSV.exists():
        print(f"{OUT_CSV} already present, skipping download (delete it to refetch)")
        return 0
    zip_bytes = download_zip()
    df = extract_transactions(zip_bytes)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=COLUMN_RENAME)
    df["CustomerID"] = pd.to_numeric(df["CustomerID"], errors="coerce")
    missing = [c for c in [*COLUMNS, "CustomerID"] if c not in df.columns]
    if missing:
        print(f"unexpected schema, missing columns: {missing}", file=sys.stderr)
        return 1
    df = df[[*COLUMNS, "CustomerID"]].copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(
        f"wrote {len(df):,} rows to {OUT_CSV} "
        f"({OUT_CSV.stat().st_size / 1e6:.1f} MB), "
        f"date range {df['InvoiceDate'].min()} .. {df['InvoiceDate'].max()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
