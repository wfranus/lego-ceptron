from argparse import ArgumentParser
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(args):
    df = pd.read_csv("./ImageSetKey.csv", dtype=str)

    img_dir = 'Base Images' if args.use == 'base' else 'Cropped Images'
    df = df[df["Folder1"] == img_dir]

    # exclude ivalid images
    # if args.use == 'cropped':
    #     invalid_images = ['1_Brick_1x1_180708133346.jpg',
    #                       '1_Brick_1x1_180708133400.jpg']
    #     df = df[~df["Name"].isin(invalid_images)]

    new_df = pd.DataFrame()
    new_df["filename"] = df["Folder1"] + "/" + df["Folder2"] + "/" + df["Name"]
    new_df["brick_type"] = df["Brick Type"]

    # Split whole set into train and test, 80% and 20% respectively
    train_df, test_df = train_test_split(new_df, test_size=0.2, random_state=42,
                                         stratify=new_df["brick_type"])
    # Split train into actual train and valid, 80% and 20% respectively
    train_df, valid_df = train_test_split(train_df, test_size=0.2, random_state=42,
                                          stratify=train_df["brick_type"])
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    train_df.to_csv("./train.csv", index=False)
    valid_df.to_csv("./valid.csv", index=False)
    test_df.to_csv("./test.csv", index=False)

    # plot class distributions
    fig, ax = plt.subplots(figsize=(12, 10))
    new_df["brick_type"].value_counts().plot(kind='barh')
    train_df["brick_type"].value_counts().plot(kind='barh', color='orange')
    test_df["brick_type"].value_counts().plot(kind='barh', color='green')
    valid_df["brick_type"].value_counts().plot(kind='barh', color='red')
    ax.invert_yaxis()
    plt.title('Class distribution')
    plt.xlabel('Number of occurrences')
    plt.ylabel('Class')
    plt.tight_layout()
    plt.legend(['total', 'train', 'test', 'valid'])
    plt.savefig('class_distribution.png')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--use', choices=('base', 'cropped'),
                        default='cropped', help='Type of images to use.')

    split_data(parser.parse_args())
