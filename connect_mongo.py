from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["trading_journal"]
plots = db["eda_plots"]

def save_plot_to_mongo(plot_name, base64_img):
    plots.insert_one({
        "plot_name": plot_name,
        "image_base64": base64_img
    })
