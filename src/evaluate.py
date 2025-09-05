import pandas as pd
import pickle
import sys

input_file = sys.argv[1]
model_file = sys.argv[2]

df = pd.read_csv(input_file)
X = df[["feature1", "feature2", "feature_sum"]]
y = df["target"]

with open(model_file, "rb") as f:
    model = pickle.load(f)

accuracy = model.score(X, y)
print("✅ Accuracy:", accuracy)
