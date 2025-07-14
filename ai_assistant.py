# ai_assistant.py
# This file integrates an LLM-powered assistant with your trading system using OpenAI API
# and connects to MongoDB/MySQL to analyze and respond to trade-related queries

import os
import openai
import pandas as pd
from pymongo import MongoClient
import mysql.connector
from datetime import datetime

# === Setup OpenAI API ===
openai.api_key = os.getenv("OPENAI_API_KEY")  # make sure to export this in .env or terminal

# === Connect to MongoDB ===
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["trading_journal"]
notes_col = mongo_db["trade_notes"]
ai_outputs_col = mongo_db["ai_outputs"]

# === Connect to MySQL ===
mysql_conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="your_password",
    database="trading"
)
mysql_cursor = mysql_conn.cursor(dictionary=True)

# === Query Helper Function ===
def fetch_mysql(query):
    mysql_cursor.execute(query)
    return mysql_cursor.fetchall()

# === LLM Query Function ===
def ask_llm(question, context):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # can be swapped later to WatsonX or LLaMA if hosted
        messages=[
            {"role": "system", "content": "You are a trading assistant who answers questions using context below."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ]
    )
    return response.choices[0].message["content"]

# === Main Analysis Function ===
def handle_user_query(question):
    context_query = """
        SELECT symbol, pnl, max_drawdown, rr_ratio, trade_date 
        FROM trades 
        WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH);
    """
    rows = fetch_mysql(context_query)
    df = pd.DataFrame(rows)
    context = df.to_string(index=False)
    answer = ask_llm(question, context)

    # Log to MongoDB
    ai_outputs_col.insert_one({
        "timestamp": datetime.utcnow(),
        "question": question,
        "response": answer,
        "context_summary": df.describe().to_dict()
    })

    print("\n--- AI Response ---\n")
    print(answer)

# === Example Query ===
if __name__ == "__main__":
    q = input("Ask a trading question: ")
    handle_user_query(q)
