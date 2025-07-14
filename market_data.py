import argparse
import os
import sys
import requests
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# === Load environment variables from .env ===
load_dotenv()

# === Alpha Vantage API Settings ===
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"
JSON_COL_REMAP = {
    "1. open": "open",
    "2. high": "high",
    "3. low": "low",
    "4. close": "close",
    "5. volume": "volume"
}

# === MySQL Config ===
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_DB = os.getenv("MYSQL_DB")


def fetch_intraday(symbol, interval, outputsize="compact", month=None,
                   adjusted="true", extended_hours="true", datatype="json"):

    params = {
        "function": "TIME_SERIES_INTRADAY",
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": API_KEY,
        "adjusted": adjusted,
        "extended_hours": extended_hours,
        "datatype": datatype
    }
    if month:
        params["month"] = month

    resp = requests.get(BASE_URL, params=params, timeout=30)
    data = resp.json()

    if "Error Message" in data or "Note" in data:
        raise RuntimeError(data.get("Error Message") or data.get("Note"))

    series_key = f"Time Series ({interval})"
    if series_key not in data:
        raise RuntimeError("Unexpected response format.")

    df = pd.DataFrame.from_dict(data[series_key], orient="index")
    df.rename(columns=JSON_COL_REMAP, inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df.astype(float).sort_index()
    df["symbol"] = symbol
    df["interval"] = interval
    df["timestamp"] = df.index
    df.reset_index(drop=True, inplace=True)
    return df

def save_to_mysql(df, table_name):
    engine = create_engine(f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}")
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f"✅ Saved {len(df)} rows to table `{table_name}`")

def main():
    parser = argparse.ArgumentParser(description="Fetch Alpha Vantage intraday data and save to MySQL")
    parser.add_argument("symbol")
    parser.add_argument("interval", choices=["1min", "5min", "15min", "30min", "60min"])
    parser.add_argument("--outputsize", default="compact")
    parser.add_argument("--month")
    parser.add_argument("--adjusted", choices=["true", "false"], default="true")
    parser.add_argument("--extended-hours", choices=["true", "false"], default="true")

    args = parser.parse_args()

    try:
        df = fetch_intraday(args.symbol.upper(), args.interval, args.outputsize,
                            args.month, args.adjusted, args.extended_hours)
        table_name = f"market_{args.symbol.lower()}_{args.interval}"
        save_to_mysql(df, table_name)
    except Exception as e:
        sys.exit(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
