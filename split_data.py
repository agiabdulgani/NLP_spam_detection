from pathlib import Path
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

CLEAN_CSV_PATH = Path("data/processed/dataset_judol_clean.csv")
SPLIT_DIR = Path("data/splits")
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

df = pd.read_csv(CLEAN_CSV_PATH)
df["group_id"] = df["group_id"].astype(str).str.strip()
df["doc_hash"] = df["doc_hash"].astype(str).str.strip()

gss_test = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
trainval_idx, test_idx = next(gss_test.split(df, groups=df["group_id"]))

df_trainval = df.iloc[trainval_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

gss_val = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
train_idx, val_idx = next(gss_val.split(df_trainval, groups=df_trainval["group_id"]))

df_train = df_trainval.iloc[train_idx].reset_index(drop=True)
df_val = df_trainval.iloc[val_idx].reset_index(drop=True)

df_train.to_csv(SPLIT_DIR / "train.csv", index=False, encoding="utf-8-sig")
df_val.to_csv(SPLIT_DIR / "val.csv", index=False, encoding="utf-8-sig")
df_test.to_csv(SPLIT_DIR / "test.csv", index=False, encoding="utf-8-sig")

splits = {"train": df_train, "val": df_val, "test": df_test}
for name, part in splits.items():
    print(f"[{name}] baris={len(part)} | grup_unik={part['group_id'].nunique()} | label={part['label'].value_counts().to_dict()}")

groups_train = set(df_train["group_id"])
groups_val = set(df_val["group_id"])
groups_test = set(df_test["group_id"])

hashes_train = set(df_train["doc_hash"])
hashes_val = set(df_val["doc_hash"])
hashes_test = set(df_test["doc_hash"])

print("\n=== AUDIT ANTI-LEAKAGE ===")
print("Overlap group_id (train & val) :", len(groups_train & groups_val))
print("Overlap group_id (train & test):", len(groups_train & groups_test))
print("Overlap group_id (val & test)  :", len(groups_val & groups_test))
print("Overlap doc_hash (train & val) :", len(hashes_train & hashes_val))
print("Overlap doc_hash (train & test):", len(hashes_train & hashes_test))
print("Overlap doc_hash (val & test)  :", len(hashes_val & hashes_test))
