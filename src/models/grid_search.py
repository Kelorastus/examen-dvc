import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()
param_grid = {
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth": [3, 5],
}
grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid)
grid.fit(X_train, y_train)
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(grid.best_params_, f)
