"""Train a TF-IDF + logistic regression sentiment classifier."""
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

df = pd.read_csv("reviews.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), lowercase=True),
    LogisticRegression(max_iter=1000),
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(f"Test accuracy: {accuracy_score(y_test, pred):.2f}")
print(classification_report(y_test, pred))
joblib.dump(model, "model.joblib")
print("Saved model.joblib")
