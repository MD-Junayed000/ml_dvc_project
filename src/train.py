import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
import sys

input_file = sys.argv[1]
model_file = sys.argv[2]

df = pd.read_csv(input_file)
X = df[["feature1", "feature2", "feature_sum"]]
y = df["target"]

model = LogisticRegression().fit(X, y)

with open(model_file, "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved:", model_file)
