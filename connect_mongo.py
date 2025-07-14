from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["trading_journal"]
notes = db["trade_notes"]
collection = db["ai_outputs"]                       # Create or access collection

notes.insert_one({
    "trade_id": 101,
    "ticker": "TSLA",
    "note": "Could've exited earlier. RSI was overbought.",
    "ai_feedback": "Next time, consider partial exits at resistance levels."
})



# --- Step 2: Insert Data (example) ---
example_output = {
    "trade_id": "TSLA_20250712_1",
    "entry_price": 220.5,
    "exit_price": 229.3,    
    "pnl": 8.8,
    "rr_ratio": 2.5,
    "model_prediction": "profitable",
    "explanation": "Entry aligned with MACD crossover + volume surge."
}

collection.insert_one(example_output)
print("✅ Inserted record into MongoDB.")
