from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def build_model():
    return RandomForestRegressor(n_estimators=100 , random_state=42)
def main() :

    train_df = pd.read_csv('data/processed/train.csv')
    x_train = train_df.drop("MedHouseVal" , axis=1)   # axis =  1 remove column
    y_train = train_df['MedHouseVal']
    model =  build_model()
    print("Model is training.")
    model.fit(x_train , y_train)
    print("Model trained successfully.")
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model , "models/rf_regression.pkl")
    print("Model saved.")
if __name__ == "__main__":
    main()

