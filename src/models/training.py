import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()
with open("models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)
model = GradientBoostingRegressor(**best_params, random_state=42)
model.fit(X_train, y_train)
with open("models/gbr_model.pkl", "wb") as f:
    pickle.dump(model, f)
