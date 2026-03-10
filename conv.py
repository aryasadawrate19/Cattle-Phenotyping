import pandas as pd

df = pd.read_excel("measurements.xlsx")

labels = pd.DataFrame({
    "image_name": df["Num"].astype(str) + ".png",
    "weight": df["Body weight (kg)"],
    "bcs": 3.0
})

labels.to_csv("labels.csv", index=False)

print("labels.csv created")