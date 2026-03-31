import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

ticker = 'AAPL'
data = yf.download(ticker, start='2020-01-01', end='2024-01-01')

df = data[['Close']].copy()
df['Target'] = df['Close'].shift(-1)
df['Lag_1'] = df['Close'].shift(1)
df['Lag_2'] = df['Close'].shift(2)
df['Lag_3'] = df['Close'].shift(3)
df['MA_5'] = df['Close'].rolling(window=5).mean()

df.dropna(inplace=True)

X = df[['Lag_1', 'Lag_2', 'Lag_3', 'MA_5']]
y = df['Target']

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

plt.figure(figsize=(14, 7))
plt.plot(df.index[split:], y_test, label='Actual Price', color='royalblue', lw=2)
plt.plot(df.index[split:], predictions, label='Predicted Price', color='darkorange', linestyle='--')
plt.title(f'{ticker} Stock Price Prediction (Random Forest)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Root Mean Squared Error: {rmse:.2f}")





