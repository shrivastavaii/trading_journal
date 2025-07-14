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
from data_processor import clean_and_save_to_mysql


# Initialize 
load_dotenv()
st.set_page_config(layout="wide", page_title="Trading Journal")

# Database Connections 
def get_mysql_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DB", "trading")
    )

# DeepSeek AI Query 
def query_deepseek(prompt):
    headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        json={"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]},
        headers=headers,
        timeout=30
    )
    return response.json()["choices"][0]["message"]["content"]

# ML Analysis 
def run_ml_analysis(df):
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    df['profitable'] = (df['pnl'] > 0).astype(int)
    X = df[['entry_price', 'stop_loss', 'quantity']].fillna(0)
    y = df['profitable']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    lr_acc = lr_model.score(X_test, y_test)
    
    xgb_model = XGBClassifier()
    xgb_model.fit(X_train, y_train)
    xgb_acc = xgb_model.score(X_test, y_test)
    
    kmeans = KMeans(n_clusters=3)
    df['cluster'] = kmeans.fit_predict(df[['pnl', 'rr_ratio']])
    
    return lr_acc, xgb_acc, df




# Market Data Fetching 
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

# Streamlit UI 
st.title("📈 Trading Journal")
st.markdown("Upload trades, fetch market data, and analyze with AI")


# File Upload Section 
with st.expander("📤 Upload Trade Data", expanded=True):
    uploaded_file = st.file_uploader("Drag & Drop Webull/CSV File", type=["csv"])
   
if uploaded_file:
    with st.spinner("Processing..."):
        try:
            # Process and store in MySQL
            df_cleaned = clean_and_save_to_mysql(tmp_path)
            
            # Store in MongoDB
            from eda_visuals import generate_and_store_plots
            plots = generate_and_store_plots(df_cleaned)
            
            # Also store the raw trades data in MongoDB
            client = MongoClient("mongodb://localhost:27017/")
            db = client["trading_journal"]
            db["trades"].insert_many(df_cleaned.to_dict('records'))
            
            st.success("✅ Data stored in both MySQL and MongoDB")
        except Exception as e:
            st.error(f"Storage error: {str(e)}")


# AI Chat Section
st.header("💬 Ask Your Trading Assistant")
question = st.text_input(
    "Type your question (e.g., 'What was my best trade last week?')",
    disabled=not st.session_state.get("data_loaded", True)
)

if st.button("Get AI Analysis") and question:
    with st.spinner("Analyzing..."):
        conn = get_mysql_connection()
        df = pd.read_sql("SELECT * FROM trades", conn)
        
        if df.empty:
            st.warning("No data found. Upload a CSV first!")
        else:
            prompt = f"""
            Analyze these trades (last 5 rows shown):
            {df.tail().to_dict('records')}
            
            Question: {question}
            
            Answer with:
            1. Key statistics
            2. Actionable insights
            3. Suggested improvements
            """
            answer = query_deepseek(prompt)
            st.success(answer)
            
            # to MongoDB
            MongoClient("mongodb://localhost:27017/")["trading_journal"]["ai_chats"].insert_one({
                "question": question,
                "answer": answer,
                "timestamp": datetime.utcnow()
            })


# Market Data Fetch Section 
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

# Analysis Tabs 
tab1, tab2, tab3 = st.tabs(["📊 Visualizations", "🤖 ML Insights", "📡 Market Data"])


# EDA
with tab1:
    st.header("Exploratory Analysis")
    conn = get_mysql_connection()
    df_eda = pd.read_sql("SELECT * FROM trades", conn)
    
    if not df_eda.empty:
        from eda_visuals import generate_eda_plots
        
        # Generate all plots
        plots = generate_eda_plots(df_eda)
        
        # Display plots in a structured way
        st.plotly_chart(plots["equity_curve"], use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plots["total_pnl"], use_container_width=True)
        with col2:
            st.plotly_chart(plots["profit_dist"], use_container_width=True)
        
        st.plotly_chart(plots["win_rate"], use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            st.plotly_chart(plots["trade_duration"], use_container_width=True)
        with col4:
            st.plotly_chart(plots["heatmap"], use_container_width=True)
    else:
        st.warning("Upload data to see visualizations")


# Tab 2: ML
with tab2:
    st.header("Machine Learning Analysis")
    conn = get_mysql_connection()
    df_ml = pd.read_sql("SELECT * FROM trades", conn)
    
    if not df_ml.empty:
        if st.button("Run Analysis"):
            with st.spinner("Training models..."):
                lr_acc, xgb_acc, df_clustered = run_ml_analysis(df_ml)
                
                st.metric("Model Accuracy (Logistic Regression)", f"{lr_acc:.1%}")
                st.metric("Model Accuracy (XGBoost)", f"{xgb_acc:.1%}")
                
                st.plotly_chart(px.scatter(
                    df_clustered, x='pnl', y='rr_ratio',
                    color='cluster', hover_data=['symbol'],
                    title="Trade Clusters by PnL and Risk-Reward"
                ))
    else:
        st.warning("Upload data to enable ML analysis")


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