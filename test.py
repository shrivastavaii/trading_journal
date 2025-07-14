


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

# --- ML Analysis ---
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

# --- Streamlit UI ---
st.title("📈 Trading Journal AI")
st.markdown("Upload your trade data and analyze performance with AI")

# --- File Upload Section (Homepage) ---
with st.expander("📤 Upload Trade Data", expanded=True):
    uploaded_file = st.file_uploader(
        "Drag & Drop Webull/CSV File", 
        type=["csv"],
        help="Supported formats: Webull, TradingView, or custom CSV with columns like 'Symbol', 'PnL'"
    )
    
    if uploaded_file:
        with st.spinner("Processing data..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                df_clean = clean_and_upload(tmp_path)
                st.success(f"✅ Success! Processed {len(df_clean)} trades.")
                st.dataframe(df_clean.head(3))
                os.unlink(tmp_path)
                st.rerun()  # Refresh other tabs
            except Exception as e:
                st.error(f"Error: {str(e)}")
                os.unlink(tmp_path)

# --- AI Chat Section (Homepage) ---
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
            
            # Log to MongoDB
            MongoClient("mongodb://localhost:27017/")["trading_journal"]["ai_chats"].insert_one({
                "question": question,
                "answer": answer,
                "timestamp": datetime.utcnow()
            })

# --- Analysis Tabs ---
tab1, tab2 = st.tabs(["📊 Visualizations", "🤖 ML Insights"])

# Tab 1: EDA
with tab1:
    st.header("Exploratory Analysis")
    conn = get_mysql_connection()
    df_eda = pd.read_sql("SELECT * FROM trades", conn)
    
    if not df_eda.empty:
        df_eda['trade_date'] = pd.to_datetime(df_eda['trade_date'])
        df_eda['day'] = df_eda['trade_date'].dt.day_name()
        df_eda['cumulative_pnl'] = df_eda['pnl'].cumsum()
        
        st.plotly_chart(px.line(df_eda, x='trade_date', y='cumulative_pnl', 
                              title="Equity Curve Over Time"))
        
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.bar(df_eda.groupby('symbol')['pnl'].sum().reset_index(), 
                                 x='symbol', y='pnl', title="Total PnL by Symbol"))
        with col2:
            st.plotly_chart(px.pie(df_eda, names='symbol', values='pnl', 
                                 title="Profit Distribution"))
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



