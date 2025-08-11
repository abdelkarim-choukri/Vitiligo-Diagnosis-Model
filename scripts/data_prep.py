# scripts/data_prep.py

import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    # 1) Load the full labels
    df = pd.read_csv("data/labels.csv")

    # 2) 10% holdout for test
    trainval_df, test_df = train_test_split(
        df,
        test_size=0.10,
        stratify=df["label"],
        random_state=42
    )

    # 3) From the remaining 90%, take ~11.11% for validation to get 10% of original
    #    (so train: 0.9*0.8889≈0.8, val:0.9*0.1111≈0.10)
    train_df, val_df = train_test_split(
        trainval_df,
        test_size=0.1111,
        stratify=trainval_df["label"],
        random_state=42
    )

    # 4) Make output folder
    os.makedirs("data/splits", exist_ok=True)

    # 5) Save CSVs
    train_df.to_csv("data/splits/train.csv", index=False)
    val_df.to_csv("data/splits/val.csv",   index=False)
    test_df.to_csv("data/splits/test.csv", index=False)

    print("✅ Created splits:")
    print(f"   train: {len(train_df)} samples")
    print(f"   val:   {len(val_df)} samples")
    print(f"   test:  {len(test_df)} samples")

if __name__ == "__main__":
    main()
