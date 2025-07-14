import os
import requests
import pandas as pd
from pymongo import MongoClient
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()  


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  

# Connect to MongoDB 
mongo_client = MongoClient("mongodb://localhost:27017/")
mongo_db = mongo_client["trading_journal"]
notes_col = mongo_db["trade_notes"]
ai_outputs_col = mongo_db["ai_outputs"]



# Connect to MySQL 
mysql_conn = mysql.connector.connect(
    host=os.getenv("MYSQL_HOST", "localhost"), 
    user=os.getenv("MYSQL_USER"),              
    password=os.getenv("MYSQL_PASSWORD"),       
    database=os.getenv("MYSQL_DB", "trading")   
)
mysql_cursor = mysql_conn.cursor(dictionary=True)


# Query Helper Function 
def fetch_mysql(query):
    mysql_cursor.execute(query)
    return mysql_cursor.fetchall()


# DeepSeek Query Function 
def ask_deepseek(question, context):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a trading assistant who answers questions using the provided context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }
    
    try:
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error querying DeepSeek API: {str(e)}"


# Main Analysis Function 
def handle_user_query(question):
    context_query = """
        SELECT symbol, pnl, max_drawdown, rr_ratio, trade_date 
        FROM trades 
        WHERE trade_date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH);
    """
    rows = fetch_mysql(context_query)
    df = pd.DataFrame(rows)
    context = df.to_string(index=False)
    answer = ask_deepseek(question, context)

    # Log to MongoDB
    ai_outputs_col.insert_one({
        "timestamp": datetime.utcnow(),
        "question": question,
        "response": answer,
        "context_summary": df.describe().to_dict()
    })

    print("\n--- AI Response ---\n")
    print(answer)


# Example Query 
if __name__ == "__main__":
    q = input("Ask a trading question: ")
    handle_user_query(q)