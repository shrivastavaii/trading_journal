
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load data
df = pd.read_csv("Webull_Trades_With_Metrics.csv")
df['trade_date'] = pd.to_datetime(df['trade_date'])
df['day'] = df['trade_date'].dt.day_name()
df = df.sort_values('trade_date')
df['cumulative_pnl'] = df['pnl'].cumsum()

# 1. Total PnL by Ticker
plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='symbol', y='pnl', estimator=sum, ci=None)
plt.title("Total Profit and Loss by Ticker")
plt.ylabel("Total PnL ($)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Equity Curve
plt.figure(figsize=(10, 5))
plt.plot(df['trade_date'], df['cumulative_pnl'], marker='o')
plt.title("Equity Curve Over Time")
plt.xlabel("Date")
plt.ylabel("Cumulative PnL ($)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Heatmap of Avg PnL by Day and Ticker
pivot = df.pivot_table(index='day', columns='symbol', values='pnl', aggfunc='mean')
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Average PnL by Day of Week and Ticker")
plt.ylabel("Day of Week")
plt.xlabel("Ticker")
plt.tight_layout()
plt.show()

# 4. Strategy-wise PnL
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='strategy', y='pnl', estimator=sum, ci=None)
plt.title("Total PnL by Strategy")
plt.ylabel("Total PnL ($)")
plt.tight_layout()
plt.show()

# 5. Interactive Equity Curve
fig = px.line(df, x='trade_date', y='cumulative_pnl', title="Interactive Equity Curve")
fig.update_traces(mode="lines+markers")
fig.show()

# 6. Interactive Bar Chart for Ticker PnL
fig2 = px.bar(df.groupby("symbol", as_index=False).sum(), x="symbol", y="pnl", title="PnL by Ticker")
fig2.show()

# 7. Win Rate by Ticker
df['win'] = df['pnl'] > 0
win_rate = df.groupby('symbol')['win'].mean().reset_index()
fig3 = px.bar(win_rate, x='symbol', y='win', title='Win Rate by Ticker')
fig3.update_yaxes(tickformat=".0%")
fig3.show()

# 8. Trade Duration Distribution
fig4 = px.histogram(df, x='time_in_trade_min', nbins=30, title='Trade Duration Distribution')
fig4.show()
