import os
import sys

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
