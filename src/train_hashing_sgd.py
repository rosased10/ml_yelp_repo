import argparse, joblib, numpy as np, pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score

ap = argparse.ArgumentParser()
ap.add_argument("--reviews_json", required=True, help="Path to yelp_academic_dataset_review.json (JSONL)")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--chunk",  type=int, default=100_000)
ap.add_argument("--val_rate", type=float, default=0.10)
ap.add_argument("--val_cap",  type=int, default=50_000)
ap.add_argument("--out_model", default="models/review_stars_hashing_sgd.joblib")
args = ap.parse_args()

vect = HashingVectorizer(n_features=2**21, ngram_range=(1,2), alternate_sign=False,
                         lowercase=True, stop_words="english", norm="l2", dtype=np.float32)
clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, average=True, random_state=42)

classes = np.array([1,2,3,4,5], dtype=int)
val_texts, val_y = [], []

for epoch in range(1, args.epochs+1):
    print(f"==== EPOCH {epoch}/{args.epochs} ====")
    processed = 0
    reader = pd.read_json(args.reviews_json, lines=True, chunksize=args.chunk)
    for chunk_df in reader:
        sub = chunk_df[["text","stars"]].dropna()
        if sub.empty: 
            continue
        # collect small validation buffer
        if len(val_texts) < args.val_cap:
            tail = int(len(sub) * args.val_rate)
            if tail > 0:
                vs = sub.iloc[-tail:]
                val_texts.extend(vs["text"].astype(str).tolist())
                val_y.extend(vs["stars"].astype(int).tolist())
                sub = sub.iloc[:-tail]
        X = vect.transform(sub["text"].astype(str))
        y = sub["stars"].astype(int).to_numpy()
        if processed == 0 and epoch == 1:
            clf.partial_fit(X, y, classes=classes)
        else:
            clf.partial_fit(X, y)
        processed += len(sub)
        if processed % 200_000 < args.chunk:
            print(f"  trained on ~{processed:,} rows...")

# validate if we have a buffer
if val_texts:
    Xv = vect.transform(val_texts)
    yv = np.array(val_y, dtype=int)
    pred = clf.predict(Xv)
    print("VAL — acc=", round(accuracy_score(yv, pred),3),
          "macro_f1=", round(f1_score(yv, pred, average='macro'),3),
          "QWK=", round(cohen_kappa_score(yv, pred, weights='quadratic'),3))
    print(classification_report(yv, pred, digits=3))

pipe = Pipeline([("vect", vect), ("clf", clf)])
joblib.dump(pipe, args.out_model)
print("Saved:", args.out_model)
