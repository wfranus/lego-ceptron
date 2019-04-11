import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("./ImageSetKey.csv", dtype=str)

new_df = pd.DataFrame()
new_df["filename"] = df["Folder1"] + "/" + df["Folder2"] + "/" + df["Name"]
new_df["brick_type"] = df["Brick Type"]

train_df, test_df = train_test_split(new_df, test_size=0.2,
                                     random_state=42, stratify=new_df["brick_type"])
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

train_df.to_csv("./train.csv", index=False)
test_df.to_csv("./test.csv", index=False)

# plot class distributions
fig, ax = plt.subplots(figsize=(12, 10))
new_df["brick_type"].value_counts().plot(kind='barh')
train_df["brick_type"].value_counts().plot(kind='barh', color='orange')
test_df["brick_type"].value_counts().plot(kind='barh', color='green')
ax.invert_yaxis()
plt.title('Class distribution')
plt.xlabel('Number of occurrences')
plt.ylabel('Class')
plt.tight_layout()
plt.legend(['total', 'train', 'test'])
plt.savefig('class_distribution.png')
