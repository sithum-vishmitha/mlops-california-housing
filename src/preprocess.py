from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    input_file  = Path("data/raw/california_housing.csv")
    df = pd.read_csv(input_file)
    train_df , test_df=  train_test_split(df , test_size = 0.2, random_state = 42)
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed_dir / "train.csv", index = False)
    test_df.to_csv(processed_dir / "test.csv", index = False)
    print("Training Set:", train_df.shape)
    print("Testing Set :", test_df.shape)

if __name__ == "__main__":
    main()