import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from prophet import Prophet

st.set_page_config(page_title="Live Stock Intelligence", layout="wide")
 
if 'live_data' not in st.session_state:
    st.session_state.live_data = []
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

st.title("Live Stock Market Intelligence")


ticker = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
update_interval = st.sidebar.slider("Update Interval (seconds)", 5, 60, 10)
alert_threshold = st.sidebar.number_input("Alert Threshold (%)", 1.0, 10.0, 2.5)

@st.cache_data(ttl=3600)
def load_historical_and_predict(ticker):
    """Load historical data and generate forecast"""
    try:
        data = yf.download(ticker, start="2023-01-01", end=datetime.now(), auto_adjust=True, progress=False)
        
        if data.empty:
            return None, None
        
        df = data.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)
        
        m = Prophet(daily_seasonality=True, changepoint_prior_scale=0.05)
        m.fit(df)
        
        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)
        
        predictions = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)
        
        return df, predictions
    
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def fetch_live_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period='1d', interval='1m')
        
        if not data.empty:
            latest = data.iloc[-1]
            return {
                'timestamp': datetime.now(),
                'price': latest['Close'],
                'volume': latest['Volume'],
                'high': latest['High'],
                'low': latest['Low']
            }
    except:
        pass
    return None

def check_alert(current_price, predicted_price, threshold):
    """Check if price crossed prediction threshold"""
    if predicted_price is None:
        return False
    
    deviation = abs((current_price - predicted_price) / predicted_price) * 100
    return deviation >= threshold

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader(f"Live Data Stream: {ticker}")
    chart_placeholder = st.empty()

with col2:
    st.subheader("Live Metrics")
    metric_placeholder = st.empty()
    alert_placeholder = st.empty()

if st.session_state.historical_data is None:
    with st.spinner("Loading historical data and generating predictions..."):
        hist_df, predictions = load_historical_and_predict(ticker)
        st.session_state.historical_data = hist_df
        st.session_state.predictions = predictions

if st.session_state.predictions is not None:
    st.sidebar.subheader("7-Day Forecast")
    pred_display = st.session_state.predictions.copy()
    pred_display['ds'] = pred_display['ds'].dt.strftime('%Y-%m-%d')
    pred_display.columns = ['Date', 'Predicted', 'Lower', 'Upper']
    st.sidebar.dataframe(pred_display.style.format({
        'Predicted': '${:.2f}',
        'Lower': '${:.2f}',
        'Upper': '${:.2f}'
    }), hide_index=True)

auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)

if st.sidebar.button("Refresh Now") or auto_refresh:
    
    live_point = fetch_live_price(ticker)
    
    if live_point:
        st.session_state.live_data.append(live_point)
        if len(st.session_state.live_data) > 100:
            st.session_state.live_data.pop(0)
        
        today = datetime.now().date()
        predicted_today = None
        if st.session_state.predictions is not None:
            pred_row = st.session_state.predictions[
                st.session_state.predictions['ds'].dt.date == today
            ]
            if not pred_row.empty:
                predicted_today = pred_row.iloc[0]['yhat']
        
        alert_triggered = check_alert(
            live_point['price'], 
            predicted_today, 
            alert_threshold
        )
        
        with metric_placeholder.container():
            m1, m2, m3 = st.columns(3)
            m1.metric("Current Price", f"${live_point['price']:.2f}")
            
            if predicted_today:
                diff = live_point['price'] - predicted_today
                m2.metric("vs Prediction", f"${diff:+.2f}", 
                         delta=f"{(diff/predicted_today)*100:+.2f}%")
            
            m3.metric("Volume", f"{live_point['volume']:,.0f}")
            
            st.caption(f"Last updated: {live_point['timestamp'].strftime('%H:%M:%S')}")
        
        if alert_triggered:
            alert_placeholder.warning(
                f" ALERT: Price deviated {alert_threshold}%+ from prediction!"
            )
        else:
            alert_placeholder.empty()
        
        fig = go.Figure()
        
        if st.session_state.historical_data is not None:
            hist_recent = st.session_state.historical_data.tail(30)
            fig.add_trace(go.Scatter(
                x=hist_recent['ds'],
                y=hist_recent['y'],
                mode='lines',
                name='Historical',
                line=dict(color='cyan', width=2)
            ))
        
        if st.session_state.predictions is not None:
            fig.add_trace(go.Scatter(
                x=st.session_state.predictions['ds'],
                y=st.session_state.predictions['yhat'],
                mode='lines',
                name='Forecast',
                line=dict(color='orange', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=st.session_state.predictions['ds'],
                y=st.session_state.predictions['yhat_upper'],
                mode='lines',
                showlegend=False,
                line=dict(width=0),
                fillcolor='rgba(255,165,0,0.2)',
                fill='tonexty'
            ))
            fig.add_trace(go.Scatter(
                x=st.session_state.predictions['ds'],
                y=st.session_state.predictions['yhat_lower'],
                mode='lines',
                showlegend=False,
                line=dict(width=0)
            ))
        
        if len(st.session_state.live_data) > 0:
            live_df = pd.DataFrame(st.session_state.live_data)
            fig.add_trace(go.Scatter(
                x=live_df['timestamp'],
                y=live_df['price'],
                mode='lines+markers',
                name='Live Data',
                line=dict(color='#FF0055', width=3),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            template='plotly_dark',
            title=f"{ticker} - Real-Time Price vs Forecast",
            xaxis_title="Time",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500
        )
        
        chart_placeholder.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Unable to fetch live data. Market might be closed or ticker invalid.")

if auto_refresh:
    time.sleep(update_interval)
    st.rerun()

with st.expander(" How It Works"):
    st.markdown("""
    **Real-Time Features:**
    - Fetches live 1-minute interval data from Yahoo Finance
    - Compares current price against AI predictions
    - Triggers alerts when price deviates beyond threshold
    - Continuously updates without page refresh
    
    **Data Sources:**
    - Historical: 2 years of daily data
    - Predictions: Prophet model with 7-day forecast
    - Live: 1-minute interval updates
    
    **Note:** Live updates work during market hours (9:30 AM - 4:00 PM EST)
    """)

st.sidebar.markdown("---")
st.sidebar.caption(" Enable Auto Refresh for continuous updates")
