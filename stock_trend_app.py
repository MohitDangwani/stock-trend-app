import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go # type: ignore
import models

# Streamlit page layout and title
st.set_page_config(page_title="Stock Trend Prediction", layout="wide")
st.title("📈 Stock Trend Prediction Web App")

# Sidebar Inputs
st.sidebar.header("Input Options")
stock_symbol = st.sidebar.text_input("Enter Stock Symbol (e.g. AAPL, GOOGL)", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2010-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Fetch stock data
@st.cache
def load_data(symbol, start, end):
    return yf.download(symbol, start=start, end=end)

df = load_data(stock_symbol, start_date, end_date)

# Adding technical indicators
df = models.add_technical_indicators(df)

# Show raw data and indicators
st.subheader(f"Stock Data for {stock_symbol.upper()} from {start_date} to {end_date}")
st.write(df.tail())  # Display last few rows of the stock data

# Interactive price chart with technical indicators
def plot_stock_data(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Closing Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='SMA 50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_20'], mode='lines', name='EMA 20'))
    fig.update_layout(
        title=f"{symbol.upper()} Closing Prices with SMA and EMA",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=True
    )
    return fig

st.plotly_chart(plot_stock_data(df, stock_symbol), use_container_width=True)

# Model Selection
st.sidebar.subheader("Model Selection")
model_option = st.sidebar.selectbox("Choose Prediction Model", ("Linear Regression", "LSTM", "ARIMA", "Prophet"))

# Display prediction results
st.subheader(f"Stock Trend Prediction using {model_option}")

if model_option == "Linear Regression":
    predicted_prices = models.linear_regression_model(df)
    st.plotly_chart(models.plot_prediction(df, predicted_prices), use_container_width=True)

elif model_option == "LSTM":
    # Allow users to tweak LSTM hyperparameters
    st.sidebar.subheader("LSTM Hyperparameters")
    time_step = st.sidebar.slider("Time Step", min_value=30, max_value=100, value=60)
    units = st.sidebar.slider("Units", min_value=10, max_value=100, value=50)
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=10, value=3)
    batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32)

    predicted_prices = models.lstm_model(df, time_step=time_step, units=units, epochs=epochs, batch_size=batch_size)
    st.plotly_chart(models.plot_prediction(df, predicted_prices), use_container_width=True)

elif model_option == "ARIMA":
    predicted_prices = models.arima_model(df)
    st.plotly_chart(models.plot_prediction(df, predicted_prices), use_container_width=True)

elif model_option == "Prophet":
    predicted_prices = models.prophet_model(df)
    st.plotly_chart(models.plot_prediction(df, predicted_prices), use_container_width=True)

# Footer
st.sidebar.markdown("Powered by [Yahoo Finance](https://finance.yahoo.com/) API")
