# data_processor.py
import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()

def clean_and_upload(input_csv_path):
    """Refactored from clean_data.py to return cleaned DataFrame."""
    # --- Load and Clean ---
    df = pd.read_csv(input_csv_path)
    
    # (Add all your cleaning logic here, e.g., price formatting, datetime conversion)
    df['Price'] = df['Price'].str.replace(r"[^\d.]", "", regex=True).astype(float)
    
    # --- Generate Metrics ---
    records = []
    for name, group in df.groupby('Name'):
        # (Add your trade metric calculations here)
        records.append({
            'symbol': group['Symbol'].iloc[0],
            'pnl': (group['Exit_Price'].mean() - group['Entry_Price'].mean()) * 100
        })
    
    # --- Upload to MySQL ---
    engine = create_engine(f"mysql+mysqlconnector://{os.getenv('MYSQL_USER')}:{os.getenv('MYSQL_PASSWORD')}@{os.getenv('MYSQL_HOST')}/{os.getenv('MYSQL_DB')}")
    final_df = pd.DataFrame(records)
    final_df.to_sql("trades", con=engine, if_exists="append", index=False)
    
    return final_df  # Return cleaned data for preview