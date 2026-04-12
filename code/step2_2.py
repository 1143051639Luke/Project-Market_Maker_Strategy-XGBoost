from xgboost import XGBClassifier
import xgboost
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import sys
from sklearn.metrics import mean_squared_error, root_mean_squared_error

datasetPath = "/Users/luke/Desktop/project_intern/result/step1_5/labeled_dataset_btc_usdt_swap_side_separated.parquet"
features = ["OBI5", "OBI25", "OBI400", "NTR_10s", "mid_std_2s", "spread_bps", "hour_of_day", "trade_flow_10s", "trade_count_10s", "cumulativeVolume_5bps"]
label = "optimal_offset_bps"
# OFFSETS_BPS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.25)

# for bid
df = pd.read_parquet(datasetPath)
df['date'] = pd.to_datetime(df['date'])
# mapping = {offset: i for i, offset in enumerate(OFFSETS_BPS)}
# df["optimal_offset_bps"] = df["optimal_offset_bps"].map(mapping)
df = df[df["side"] == "ask"]
data_train = df[df['date'] < datetime(2026, 1, 26)]
data_valid = df[~(df['date'] < datetime(2026, 1, 26))]
X_train = data_train[features]
y_train = data_train[label]
X_valid = data_valid[features]
y_valid = data_valid[label]
model = xgboost.XGBRegressor(max_depth=5, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, reg_lambda=2)
model.fit(X_train, y_train)
# print(model.classes_)
preds_train = model.predict(X_train)
preds_valid = model.predict(X_valid)
mse_train = root_mean_squared_error(y_train, preds_train)
mse_valid = root_mean_squared_error(y_valid, preds_valid)
print(f"Training Error: {mse_train:.4f}")
print(f"Validation Error: {mse_valid:.4f}")

# model.save_model("xgboost_regressor_ask.json")
