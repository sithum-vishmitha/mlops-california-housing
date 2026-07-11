from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.metrics import (mean_absolute_error , mean_squared_error , r2_score)

from math  import  sqrt

def main():
    test_df =  pd.read_csv("../data/processed/test.csv")
    x_test  = test_df.drop("MedHouseVal" , axis=1)
    y_test = test_df["MedHouseVal"]

    #loas the model
    model =  joblib.load("../models/linear_regression.pkl")

    predictions = model.predict(x_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "MSE": mean_squared_error(y_test, predictions),
        "RMSE" : sqrt(mean_squared_error(y_test, predictions)),
        "R2" : r2_score(y_test, predictions)
    }

    Path("../reports").mkdir(exist_ok=True)
    with open("../reports/metrics.json", "w") as outfile:
        json.dump(metrics , outfile , indent=4)

    print(metrics)


if __name__ == "__main__":
    main()

