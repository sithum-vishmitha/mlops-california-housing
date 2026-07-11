from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression

def main() :
    train_df = pd.read_csv('../data/processed/train.csv')
    x_train = train_df.drop("MedHouseVal" , axis=1)   # axis =  1 remove column
    y_train = train_df['MedHouseVal']
    model =  LinearRegression()
    model.fit(x_train , y_train)
    print("Model trained successfully.")
    Path("../models").mkdir(exist_ok=True)
    joblib.dump(model , "../models/linear_regression.pkl")
    print("Model saved.")
if __name__ == "__main__":
    main()

