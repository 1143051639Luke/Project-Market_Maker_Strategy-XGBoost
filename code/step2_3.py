from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from datetime import datetime
import sys

datasetPath = "/Users/luke/Desktop/project_intern/result/step1_5/labeled_dataset_btc_usdt_swap_side_separated.parquet"
features = ["OBI5", "OBI25", "OBI400", "NTR_10s", "mid_std_2s", "spread_bps", "hour_of_day", "trade_flow_10s", "trade_count_10s", "cumulativeVolume_5bps"]
label = "pnl_pos_neg"

# for bid
df = pd.read_parquet(datasetPath)
df['date'] = pd.to_datetime(df['date'])
df = df[df["side"] == "ask"]
df = df[df['optimal_pnl_bps'] != 0]
# if pnl > 0, label as 1, else label as 0
df['pnl_pos_neg'] = df['optimal_pnl_bps'].apply(lambda x: 1 if x > 0 else 0)
data_train = df[df['date'] < datetime(2026, 1, 26)]
data_valid = df[~(df['date'] < datetime(2026, 1, 26))]
X_train = data_train[features]
y_train = data_train[label]
X_valid = data_valid[features]
y_valid = data_valid[label]
model = XGBClassifier(max_depth=5, n_estimators=300, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=42, reg_lambda=2)
model.fit(X_train, y_train)
# print(model.classes_)
preds_train = model.predict(X_train)
preds_valid = model.predict(X_valid)
error_train = (preds_train != y_train).mean()
error_valid = (preds_valid != y_valid).mean()
print(f"Training Error: {error_train:.4f}")
print(f"Validation Error: {error_valid:.4f}")

model.save_model("xgboost_multiClassification_netPnL_ask.json")
