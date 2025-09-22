import argparse, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument("--sample_csv", required=True, help="CSV with columns: text, stars")
ap.add_argument("--out_model", default="models/tfidf_logreg.joblib")
args = ap.parse_args()

df = pd.read_csv(args.sample_csv).dropna(subset=["text","stars"])
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["stars"].astype(int), test_size=0.2, random_state=42, stratify=df["stars"])

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=35000, ngram_range=(1,2), min_df=5, stop_words="english")),
    ("clf",   LogisticRegression(C=3.0, class_weight="balanced", solver="liblinear", max_iter=1000, random_state=42))
])
pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)

print(classification_report(y_test, pred, digits=3))
print("accuracy:", round(accuracy_score(y_test, pred), 3),
      "macro_f1:", round(f1_score(y_test, pred, average="macro"), 3),
      "QWK:", round(cohen_kappa_score(y_test, pred, weights="quadratic"), 3))

joblib.dump(pipe, args.out_model)
print("Saved:", args.out_model)
