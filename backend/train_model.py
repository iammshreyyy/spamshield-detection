import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "spam.csv")
MODEL_PATH = os.path.join(HERE, "spam_model.joblib")

def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            "Missing backend/spam.csv. Add a labeled dataset with columns "
            "'text' and 'label' (label values: spam or ham), then run again."
        )

    df = pd.read_csv(DATA_PATH)
    required = {"text", "label"}
    if not required.issubset(df.columns):
        raise SystemExit("CSV must contain columns named: text,label")

    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df = df[df["label"].isin(["spam", "ham"])]
    df = df[df["text"] != ""]

    if df["label"].nunique() < 2 or len(df) < 20:
        raise SystemExit("Please use at least 20 labeled examples and include both spam and ham.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Accuracy:", round(accuracy_score(y_test, preds), 4))
    print(classification_report(y_test, preds, zero_division=0))
    joblib.dump(model, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")

if __name__ == "__main__":
    main()
