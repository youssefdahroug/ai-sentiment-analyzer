"""Usage: python predict.py "this phone is great" """
import sys
import joblib

model = joblib.load("model.joblib")
text = " ".join(sys.argv[1:]) or input("Review: ")
label = model.predict([text])[0]
prob = model.predict_proba([text])[0].max()
print(f"{label} ({prob:.0%} confidence)")
