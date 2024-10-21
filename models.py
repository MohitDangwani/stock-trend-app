#import numpy as np
#import pandas as pd
#from sklearn.model_selection import train_test_split
#from sklearn.linear_model import LinearRegression
#from sklearn.preprocessing import MinMaxScaler
#from keras.models import Sequential  
#from keras.layers import Dense, LSTM 
#import plotly.graph_objs as go 
#from statsmodels.tsa.arima.model import ARIMA
#from prophet import Prophet

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM # type: ignore
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet


# Linear Regression Model (same as before)
def linear_regression_model(df):
    df['Date'] = df.index.map(pd.Timestamp.timestamp)
    X = np.array(df['Date']).reshape(-1, 1)
    y = np.array(df['Close'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    predicted_df = pd.DataFrame({"Date": X_test.flatten(), "Predicted_Close": predictions})
    predicted_df.sort_values(by="Date", inplace=True)
    predicted_df['Date'] = pd.to_datetime(predicted_df['Date'], unit='s')
    return predicted_df.set_index("Date")["Predicted_Close"]

# LSTM Model with Optimized Hyperparameters
def lstm_model(df, time_step=60, units=50, epochs=3, batch_size=32):
    data = df[['Close']].values

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare training and testing data
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_dataset(data, time_step=60):
        X, Y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i:(i + time_step), 0])
            Y.append(data[i + time_step, 0])
        return np.array(X), np.array(Y)

    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # Reshape input to [samples, time steps, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # LSTM Model with optimized parameters
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=(time_step, 1)))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs)

    # Predicting
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)

    predicted_df = pd.DataFrame(predictions, columns=["Predicted_Close"])
    predicted_df.index = df.index[-len(predictions):]
    return predicted_df["Predicted_Close"]

# ARIMA Model
def arima_model(df):
    df['Close_diff'] = df['Close'].diff().dropna()
    
    # ARIMA model fitting
    model = ARIMA(df['Close_diff'].dropna(), order=(5,1,0))  # Can tune order
    model_fit = model.fit()
    
    predictions = model_fit.forecast(steps=len(df))
    predicted_df = pd.DataFrame(predictions, columns=["Predicted_Close"])
    predicted_df.index = df.index[-len(predictions):]
    
    return predicted_df["Predicted_Close"]

# Prophet Model
def prophet_model(df):
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    
    model = Prophet()
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=365)  # Forecasting for one year
    forecast = model.predict(future)
    
    forecast_df = forecast[['ds', 'yhat']].set_index('ds')
    forecast_df.columns = ["Predicted_Close"]
    
    return forecast_df["Predicted_Close"]

# Technical Indicators (SMA, EMA)
def add_technical_indicators(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    return df

# Plot Prediction Results with Plotly
def plot_prediction(df, predicted_prices):
    fig = go.Figure()

    # Plot actual closing price
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Actual Price'))

    # Plot predicted closing price
    fig.add_trace(go.Scatter(x=predicted_prices.index, y=predicted_prices, mode='lines', name='Predicted Price', line=dict(dash='dash')))

    # Update layout
    fig.update_layout(
        title="Actual vs Predicted Stock Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=True
    )

    return fig
