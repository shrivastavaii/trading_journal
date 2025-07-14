# trade_ml_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# --- Load Data ---
df = pd.read_csv("Webull_Trades_With_Metrics.csv")

# --- Create Classification Target ---
df['profitable'] = (df['pnl'] > 0).astype(int)

# --- Features & Target for Classification ---
features = ['entry_price', 'stop_loss', 'take_profit', 'quantity', 'time_in_trade_min']
X = df[features].fillna(0)
y = df['profitable']

# --- Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Standardize Features ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Logistic Regression ---
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_log = log_model.predict(X_test_scaled)
log_report = classification_report(y_test, y_pred_log)

# --- XGBoost Classifier ---
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
xgb_report = classification_report(y_test, y_pred_xgb)

# --- Save Model Reports ---
with open("model_report.txt", "w") as f:
    f.write("=== Logistic Regression ===\n")
    f.write(log_report)
    f.write("\n\n=== XGBoost ===\n")
    f.write(xgb_report)

# --- KMeans Clustering on Profitability ---
cluster_features = df[['pnl', 'rr_ratio', 'time_in_trade_min']].fillna(0)
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(cluster_features)

# --- Cluster Heatmap ---
cluster_means = df.groupby('cluster')[['pnl', 'rr_ratio', 'time_in_trade_min']].mean()
sns.heatmap(cluster_means, annot=True, cmap="YlGnBu")
plt.title("Cluster Centers Heatmap")
plt.savefig("cluster_heatmap.png")
plt.close()

# --- Save clustered sample ---
df[['symbol', 'entry_price', 'exit_price', 'pnl', 'rr_ratio', 'time_in_trade_min', 'cluster']].to_csv("clustered_trades_sample.csv", index=False)

print("✅ Analysis complete. Models trained, report and visual saved.")
