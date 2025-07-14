import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

# === Load environment variables ===
load_dotenv()
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_DB = os.getenv("MYSQL_DB", "trading_journal")

# === Connect to MySQL ===
engine = create_engine(f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}")

# --- Step 1: Load and Clean Data ---
df = pd.read_csv("Webull.csv")

# Clean Price and Avg Price
df['Price'] = df['Price'].astype(str).str.replace(r"[^\d.]", "", regex=True)
df['Price'] = pd.to_numeric(df['Price'], errors='coerce').round(2)
df['Avg Price'] = pd.to_numeric(df['Avg Price'], errors='coerce').round(2)

# Standardize datetime columns
df['Placed Time'] = pd.to_datetime(df['Placed Time'], errors='coerce')
df['Filled Time'] = pd.to_datetime(df['Filled Time'], errors='coerce')

# Extract base Ticker
df['Ticker'] = df['Symbol'].str.extract(r'([A-Z]+)')

# --- Step 2: Generate Trade Metrics ---
df['Trade Date'] = df['Placed Time'].dt.strftime("%Y-%m-%d")
grouped = df.groupby('Name')  # Do NOT modify Name before this step

records = []

for name, group in grouped:
    buy_group = group[group['Side'] == 'Buy']
    sell_group = group[group['Side'] == 'Sell']

    if buy_group.empty or sell_group.empty:
        print(f"Skipping {name} — missing Buy or Sell")
        continue

    buy = buy_group.iloc[0]
    sell = sell_group.iloc[0]

    entry_price = buy['Avg Price']
    exit_price = sell['Avg Price']
    qty = min(buy['Filled'], sell['Filled'])

    if pd.isna(entry_price) or pd.isna(exit_price):
        print(f"Skipping {name} — missing entry/exit price")
        continue

    pnl = (exit_price - entry_price) * qty
    rr_ratio = round((exit_price - entry_price) / (entry_price * 0.05), 2)  # 5% risk assumption
    drawdown = round(entry_price * 0.02, 2)  # 2% placeholder
    time_in_trade = (sell['Filled Time'] - buy['Filled Time']).total_seconds() / 60 if pd.notna(sell['Filled Time']) and pd.notna(buy['Filled Time']) else np.nan

    records.append({
        'trade_date': buy['Placed Time'].date(),
        'symbol': buy['Symbol'],
        'side': buy['Side'],
        'strategy': 'Day',
        'entry_price': entry_price,
        'exit_price': exit_price,
        'stop_loss': round(entry_price * 0.95, 2),
        'take_profit': round(entry_price * 1.10, 2),
        'quantity': qty,
        'pnl': round(pnl, 2),
        'rr_ratio': rr_ratio,
        'max_drawdown': drawdown,
        'time_in_trade_min': round(time_in_trade, 1)
    })

# --- Step 3: Output ---
final_df = pd.DataFrame(records)

# Save local CSV backup
final_df.to_csv("Webull_Trades_With_Metrics.csv", index=False)
print("✅ Saved cleaned data to Webull_Trades_With_Metrics.csv")

# --- Step 4: Upload to MySQL ---
final_df.to_sql("trades", con=engine, if_exists="append", index=False)
print(f"✅ Uploaded {len(final_df)} trades to MySQL table 'trades'")
