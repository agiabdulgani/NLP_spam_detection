from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
)

SPLIT_DIR = Path("data/splits")
ARTIFACT_DIR = Path("artifacts")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(SPLIT_DIR / "train.csv")
val_df = pd.read_csv(SPLIT_DIR / "val.csv")
test_df = pd.read_csv(SPLIT_DIR / "test.csv")

for name, d in [("train", train_df), ("val", val_df), ("test", test_df)]:
    d["text"] = d["text"].fillna("").astype(str)
    d["label"] = d["label"].astype(int)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train = vectorizer.fit_transform(train_df["text"])
X_val = vectorizer.transform(val_df["text"])
X_test = vectorizer.transform(test_df["text"])

y_train = train_df["label"]
y_val = val_df["label"]
y_test = test_df["label"]

models = {
    "naive_bayes": (
        MultinomialNB(alpha=0.5),
        "naive_bayes_model.pkl",
        "confusion_matrix_nb_test.png",
        "Confusion Matrix NB - Test Set",
    ),
    "logreg_baseline": (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "logreg_baseline.pkl",
        "confusion_matrix_test.png",
        "Confusion Matrix LogReg - Test Set",
    ),
}

for model_name, (clf, pkl_name, img_name, chart_title) in models.items():
    clf.fit(X_train, y_train)

    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)

    print(f"\n================ {model_name.upper()} ================")
    print("=== VALIDATION ===")
    print("Macro F1:", round(f1_score(y_val, val_pred, average="macro", zero_division=0), 4))
    print(classification_report(y_val, val_pred, zero_division=0))

    print("=== TEST ===")
    print("Macro F1:", round(f1_score(y_test, test_pred, average="macro", zero_division=0), 4))
    print(classification_report(y_test, test_pred, zero_division=0))

    cm_test = confusion_matrix(y_test, test_pred, labels=[0, 1])
    print("Confusion Matrix (test):\n", cm_test)

    with open(ARTIFACT_DIR / pkl_name, "wb") as f:
        pickle.dump(clf, f)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=[0, 1])
    disp.plot(cmap="Blues")
    plt.title(chart_title)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / img_name, dpi=150)
    plt.close()

# Simpan artefak vectorizer sekali saja agar konsisten untuk semua model
with open(ARTIFACT_DIR / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nArtefak tersimpan di folder artifacts/")