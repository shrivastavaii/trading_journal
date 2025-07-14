import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64
from io import BytesIO
from connect_mongo import save_plot_to_mongo

# --- Load Data ---
df = pd.read_csv("Webull_Trades_With_Metrics.csv")

# Preprocess columns
df['trade_date'] = pd.to_datetime(df['Trade Date'])
df['day'] = df['trade_date'].dt.day_name()
df['symbol'] = df['Ticker']
df['pnl'] = df['PnL']
df['strategy'] = df['Strategy']
df['time_in_trade_min'] = df['Time in Trade (min)']
df['cumulative_pnl'] = df['pnl'].cumsum()

# --- Helper function to save Matplotlib plots to MongoDB ---
def save_matplotlib_plot(fig, plot_name):
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    save_plot_to_mongo(plot_name, encoded)

# --- Plot 1: Total PnL by Ticker ---
fig1 = plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='symbol', y='pnl', estimator=sum, ci=None)
plt.title("Total Profit and Loss by Ticker")
plt.ylabel("Total PnL ($)")
plt.xticks(rotation=45)
plt.tight_layout()
save_matplotlib_plot(fig1, "Total PnL by Ticker")
plt.close()

# --- Plot 2: Equity Curve ---
fig2 = plt.figure(figsize=(10, 5))
plt.plot(df['trade_date'], df['cumulative_pnl'], marker='o')
plt.title("Equity Curve Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL ($)")
plt.grid(True)
plt.tight_layout()
save_matplotlib_plot(fig2, "Equity Curve")
plt.close()

# --- Plot 3: Heatmap of Avg PnL by Day and Ticker ---
pivot = df.pivot_table(index='day', columns='symbol', values='pnl', aggfunc='mean')
fig3 = plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Average PnL by Day of Week and Ticker")
plt.ylabel("Day of Week")
plt.xlabel("Ticker")
plt.tight_layout()
save_matplotlib_plot(fig3, "Heatmap of Avg PnL")
plt.close()

# --- Plot 4: Strategy-wise PnL ---
fig4 = plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='strategy', y='pnl', estimator=sum, ci=None)
plt.title("Total PnL by Strategy")
plt.ylabel("Total PnL ($)")
plt.tight_layout()
save_matplotlib_plot(fig4, "Strategy-wise PnL")
plt.close()

# --- Plot 5: Interactive Equity Curve ---
fig5 = px.line(df, x='trade_date', y='cumulative_pnl', title="Interactive Equity Curve")
fig5.write_html("outputs/interactive_equity_curve.html")

# --- Plot 6: Interactive Bar Chart for Ticker PnL ---
fig6 = px.bar(df.groupby("symbol", as_index=False).sum(), x="symbol", y="pnl", title="PnL by Ticker")
fig6.write_html("outputs/interactive_ticker_pnl.html")

# --- Plot 7: Win Rate by Ticker ---
df['win'] = df['pnl'] > 0
win_rate = df.groupby('symbol')['win'].mean().reset_index()
fig7 = px.bar(win_rate, x='symbol', y='win', title='Win Rate by Ticker')
fig7.update_yaxes(tickformat=".0%")
fig7.write_html("outputs/win_rate_by_ticker.html")

# --- Plot 8: Trade Duration Distribution ---
fig8 = px.histogram(df, x='time_in_trade_min', nbins=30, title='Trade Duration Distribution')
fig8.write_html("outputs/trade_duration_distribution.html")

print("✅ All plots generated and saved (MongoDB or HTML where applicable).")
