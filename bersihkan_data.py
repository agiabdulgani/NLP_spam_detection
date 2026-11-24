import hashlib
import re
from pathlib import Path
import pandas as pd

RAW_CSV_PATH = Path("data/raw/reviews_raw_id.csv")
CLEAN_CSV_PATH = Path("data/processed/dataset_judol_clean.csv")
CLEAN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

JUDOL_SPAM_PATTERN = re.compile(
    r"(slot|gacor|jud[io0]|maxwin|wd|withdraw|deposit|depo|zeus|olympus|pragmatic|"
    r"rtp|bandar|togel|parlay|casino|poker|bola88|qq|judi|bantuan\s*dana|pinjol|"
    r"wa\.me|t\.me|bit\.ly|tinyurl|http[s]?://|www\.)",
    flags=re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


df = pd.read_csv(RAW_CSV_PATH)

df["text"] = df["text"].astype(str).str.strip()
df["group_id"] = df["group_id"].astype(str).str.strip()
df = df[(df["text"] != "") & (df["text"].str.lower() != "nan") & (df["group_id"] != "")].copy()

df["norm_text"] = df["text"].apply(normalize_text)
df["doc_hash"] = df["norm_text"].apply(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())

before_dedup = len(df)
df = df.drop_duplicates(subset=["doc_hash"]).reset_index(drop=True)

df["label"] = df["norm_text"].apply(lambda s: 1 if JUDOL_SPAM_PATTERN.search(s) else 0)

df[["text", "group_id", "label", "norm_text", "doc_hash", "score", "at", "source_app"]].to_csv(
    CLEAN_CSV_PATH, index=False, encoding="utf-8-sig"
)

print(f"Baris awal valid: {before_dedup} -> sesudah deduplikasi: {len(df)}")
print(f"Jumlah grup unik (group_id): {df['group_id'].nunique()}")
print("Distribusi label draf (0=normal, 1=spam/judol):")
print(df["label"].value_counts())
print("\nContoh baris terdeteksi label=1:")
print(df[df["label"] == 1][["group_id", "text", "label"]].head(5))
