import os
import sys
import pandas as pd

# --- DEBUG: show where we are ---
print("CWD:", os.getcwd())
print("__file__:", __file__)

# --- Make sure Python can see the src/ folder ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")

print("BASE_DIR:", BASE_DIR)
print("SRC_DIR:", SRC_DIR, "exists:", os.path.isdir(SRC_DIR))

# Add src/ to the Python path
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Now we can import directly from modules inside src/
from data_loader import load_price_data
from features import build_feature_set
from model import MLModel

# 1. Load raw price data
df_prices = load_price_data("AAPL", period="2y", interval="1d")

# 2. Build features + target
df_features = build_feature_set(df_prices)

# 3. Define which columns are features
feature_cols = ["Close", "SMA_5", "SMA_20", "Vol_10", "RSI_14"]

# 4. Create and train model
model = MLModel(feature_cols=feature_cols, train_ratio=0.7)
train_acc, test_acc = model.fit(df_features)

print(f"Train accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}")

# 5. Get latest signal
signal = model.predict_latest(df_features)
print("Latest model signal:", "📈 BUY/LONG" if signal == 1 else "📉 FLAT / NO-LONG")

# --- 6. Simple backtest on test period only ---

import matplotlib.pyplot as plt

# Recreate the same train/test split index
split_idx = int(len(df_features) * model.train_ratio)

test = df_features.iloc[split_idx:].copy()

# Generate signals on the test set
test["Signal"] = model.model.predict(test[feature_cols])

# Position applies to the NEXT bar
test["Position"] = test["Signal"].shift(1).fillna(0)

# Strategy returns: position * actual return
test["Strategy_Return"] = test["Position"] * test["Return"]

# Equity curves
test["BuyHold_Equity"] = (1 + test["Return"]).cumprod()
test["Strategy_Equity"] = (1 + test["Strategy_Return"]).cumprod()

# Simple max drawdown helper
def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min())

buyhold_final = test["BuyHold_Equity"].iloc[-1]
strategy_final = test["Strategy_Equity"].iloc[-1]

buyhold_dd = max_drawdown(test["BuyHold_Equity"])
strategy_dd = max_drawdown(test["Strategy_Equity"])

print("\n--- Backtest (test period only) ---")
print("Buy & Hold final equity: ", round(buyhold_final, 3))
print("Strategy final equity:   ", round(strategy_final, 3))
print("Buy & Hold max drawdown: ", f"{buyhold_dd:.1%}")
print("Strategy max drawdown:   ", f"{strategy_dd:.1%}")

# Plot equity curves
ax = test[["BuyHold_Equity", "Strategy_Equity"]].plot(
    title=f"{'AAPL'} – Buy & Hold vs ML Strategy (Test)"
)
ax.set_ylabel("Equity (starting at 1.0)")
plt.show()

