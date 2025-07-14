# web_app.py
import os
import tempfile
import requests
import pandas as pd
import streamlit as st
import mysql.connector
import plotly.express as px
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from data_processor import clean_and_upload

# --- Initialize ---
load_dotenv()
st.set_page_config(layout="wide", page_title="Trading Journal AI")

# --- Database Connections ---
def get_mysql_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB", "trading")
    )

# --- DeepSeek AI Query ---
def query_deepseek(prompt):
    headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]},
        headers=headers,
        timeout=30
    )
    return response.json()["choices"][0]["message"]["content"]

# --- Market Data Fetching ---
def fetch_market_data(symbol, interval="1day"):
    """Fetch stock data using Alpha Vantage API"""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}&datatype=csv"
    
    try:
        df = pd.read_csv(url)
        if "timestamp" not in df.columns:
            st.error("Invalid API response. Check symbol or try again later.")
            return None
        
        df.rename(columns={
            "timestamp": "date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume"
        }, inplace=True)
        
        return df.sort_values("date", ascending=False)
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

# --- Streamlit UI ---
st.title("📈 Trading Journal AI")
st.markdown("Upload trades, fetch market data, and analyze with AI")

# --- File Upload Section ---
with st.expander("📤 Upload Trade Data", expanded=True):
    uploaded_file = st.file_uploader("Drag & Drop Webull/CSV File", type=["csv"])
    if uploaded_file:
        with st.spinner("Processing..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                df_clean = clean_and_upload(tmp_path)
                st.success(f"✅ Processed {len(df_clean)} trades")
                st.dataframe(df_clean.head(3))
                os.unlink(tmp_path)
                st.session_state.data_loaded = True
            except Exception as e:
                st.error(f"Error: {str(e)}")
                os.unlink(tmp_path)

# --- Market Data Fetch Section ---
with st.expander("📡 Fetch Market Data", expanded=True):
    col1, col2 = st.columns([3,1])
    with col1:
        symbol = st.text_input("Stock Symbol (e.g., TSLA)", value="TSLA").upper()
    with col2:
        interval = st.selectbox("Interval", ["1day", "5min", "15min", "60min"])
    
    if st.button("Fetch Data"):
        with st.spinner(f"Loading {symbol} data..."):
            df_market = fetch_market_data(symbol, interval)
            if df_market is not None:
                st.session_state.market_data = df_market
                st.success(f"✅ Fetched {len(df_market)} rows")
                
                # Preview
                st.dataframe(df_market.head(), height=200)
                
                # Quick plot
                st.plotly_chart(px.line(
                    df_market.head(100), 
                    x="date", y="Close", 
                    title=f"{symbol} Price (Last 100 Periods)"
                ))

# --- Analysis Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🤖 ML Insights", "📡 Market Data"])

# Tab 1: EDA
with tab1:
    st.header("Trade Analysis")
    conn = get_mysql_connection()
    df_eda = pd.read_sql("SELECT * FROM trades", conn)
    
    if not df_eda.empty:
        df_eda['trade_date'] = pd.to_datetime(df_eda['trade_date'])
        df_eda['cumulative_pnl'] = df_eda['pnl'].cumsum()
        
        st.plotly_chart(px.line(
            df_eda, x="trade_date", y="cumulative_pnl",
            title="Your Trading Performance"
        ))
    else:
        st.warning("Upload trade data first")

# Tab 2: ML
with tab2:
    st.header("Machine Learning")
    # ... (your existing ML code) ...

# Tab 3: Market Data Explorer
with tab3:
    st.header("Historical Market Data")
    
    if "market_data" in st.session_state:
        df = st.session_state.market_data
        
        # Data Controls
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input("Date Range", [
                pd.to_datetime(df["date"].min()),
                pd.to_datetime(df["date"].max())
            ])
        with col2:
            selected_cols = st.multiselect(
                "Columns to Show", 
                df.columns,
                default=["date", "Open", "Close", "Volume"]
            )
        
        # Filtered Data
        filtered_df = df[
            (pd.to_datetime(df["date"]) >= pd.to_datetime(date_range[0])) &
            (pd.to_datetime(df["date"]) <= pd.to_datetime(date_range[1]))
        ][selected_cols]
        
        st.dataframe(filtered_df, height=300)
        
        # Interactive Chart
        st.plotly_chart(px.line(
            filtered_df, 
            x="date", y=selected_cols[1:],
            title=f"{symbol} Historical Data"
        ))
    else:
        st.info("Fetch market data from the homepage first")