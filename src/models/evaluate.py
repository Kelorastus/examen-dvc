import json
import pickle

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()
with open("models/gbr_model.pkl", "rb") as f:
    model = pickle.load(f)
y_pred = model.predict(X_test)
pd.DataFrame({"prediction": y_pred}).to_csv("data/prediction.csv", index=False)
scores = {"mse": mean_squared_error(y_test, y_pred), "r2": r2_score(y_test, y_pred)}
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f)
