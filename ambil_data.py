from pathlib import Path
import pandas as pd
from google_play_scraper import Sort, reviews_all

APP_ID = "com.disney.disneyplus"
RAW_CSV_PATH = Path("data/raw/reviews_raw_id.csv")
RAW_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

raw_reviews = reviews_all(
    APP_ID,
    sleep_milliseconds=200,
    lang="id",
    country="id",
    sort=Sort.NEWEST,
    filter_score_with=None,
)

df_raw = pd.DataFrame(raw_reviews)

df_export = pd.DataFrame({
    "text": df_raw.get("content", ""),
    "group_id": df_raw.get("userName", "").astype(str),
    "score": df_raw.get("score", None),
    "at": df_raw.get("at", None),
    "source_app": APP_ID,
    "label": "",
})

df_export.to_csv(RAW_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"Berhasil menyimpan {len(df_export)} baris ke {RAW_CSV_PATH}")
print(df_export.head(5))
