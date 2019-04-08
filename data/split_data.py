import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./ImageSetKey.csv", dtype=str)

new_df = pd.DataFrame()
new_df["filename"] = df["Folder1"] + "/" + df["Folder2"] + "/" + df["Name"]
new_df["brick_type"] = df["Brick Type"]

train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42)

train_df.to_csv("./train.csv", index=False)
test_df.to_csv("./test.csv", index=False)
