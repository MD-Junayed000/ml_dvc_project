import pandas as pd
import sys

input_file = sys.argv[1]
output_file = sys.argv[2]

df = pd.read_csv(input_file)
df["feature_sum"] = df["feature1"] + df["feature2"]
df.to_csv(output_file, index=False)
print("✅ Preprocessing done:", output_file)
