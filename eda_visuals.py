# eda_visuals.py (modified)
import pandas as pd
import plotly.express as px
import streamlit as st

def generate_eda_plots(df):
    
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df['day'] = df['trade_date'].dt.day_name()
    df['cumulative_pnl'] = df['pnl'].cumsum()
    df['win'] = df['pnl'] > 0

    

    # Plot 1: Equity Curve
    fig1 = px.line(df, x='trade_date', y='cumulative_pnl', 
                  title="Equity Curve Over Time")
    

    # Plot 2: Total PnL by Ticker
    fig2 = px.bar(df.groupby('symbol', as_index=False)['pnl'].sum(), 
                 x='symbol', y='pnl', 
                 title="Total PnL by Symbol")
    

    # Plot 3: Profit Distribution
    fig3 = px.pie(df, names='symbol', values='pnl', 
                 title="Profit Distribution")
    

    # Plot 4: Win Rate by Ticker
    win_rate = df.groupby('symbol')['win'].mean().reset_index()
    fig4 = px.bar(win_rate, x='symbol', y='win', 
                 title='Win Rate by Ticker')
    fig4.update_yaxes(tickformat=".0%")
    

    # Plot 5: Trade Duration Distribution
    fig5 = px.histogram(df, x='time_in_trade_min', nbins=30, 
                       title='Trade Duration Distribution')
    
    
    # Plot 6: Heatmap of Avg PnL by Day and Ticker
    pivot = df.pivot_table(index='day', columns='symbol', 
                          values='pnl', aggfunc='mean')
    fig6 = px.imshow(pivot, 
                    labels=dict(x="Ticker", y="Day of Week", color="PnL"),
                    x=pivot.columns,
                    y=pivot.index,
                    title="Average PnL by Day of Week and Ticker")
    
    return {
        "equity_curve": fig1,
        "total_pnl": fig2,
        "profit_dist": fig3,
        "win_rate": fig4,
        "trade_duration": fig5,
        "heatmap": fig6
    }
