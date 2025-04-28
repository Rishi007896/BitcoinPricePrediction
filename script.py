import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow import keras  # Or from keras.models import Sequential, etc.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Fetch Bitcoin price data
btc_ticker = yf.Ticker("BTC-USD")
btc_data = btc_ticker.history(period="max")
btc_data.reset_index(inplace=True)
btc_data = btc_data[['Date', 'Close']]
btc_data['Date'] = pd.to_datetime(btc_data['Date']).dt.tz_localize(None)

# Display the latest data
print(btc_data.tail())

# Load sentiment data
actual_sentiment_data = pd.read_csv("daily_sentiment_data.csv")
actual_sentiment_data['date'] = pd.to_datetime(actual_sentiment_data['date'])
actual_sentiment_data.rename(columns={"Date": "date", "SentimentScore": "sentiment_score"}, inplace=True)

btc_data_with_sentiment = pd.merge(btc_data, actual_sentiment_data, left_on='Date', right_on='date', how='left')
btc_data_with_sentiment.drop(columns=['date'], inplace=True)
btc_data_with_sentiment['sentiment_score'].fillna(0, inplace=True)

# Forward fill NaN values in 'sentiment_score'
btc_data_with_sentiment['sentiment_score'].ffill(inplace=True)  # Or .bfill() for backward fill

# If you want 0 for the initial missing values
btc_data_with_sentiment['sentiment_score'].fillna(0, inplace=True)

print(btc_data_with_sentiment.tail())

# Prepare features and target
merged_data = btc_data_with_sentiment  # Define merged_data here

features_with_sentiment = merged_data[["Close", "sentiment_score"]]
features_without_sentiment = merged_data[["Close"]]
target = merged_data["Close"].shift(-1).dropna()

features_with_sentiment = features_with_sentiment.iloc[:-1]
features_without_sentiment = features_without_sentiment.iloc[:-1]

X_train_with_sentiment, X_test_with_sentiment, y_train, y_test = train_test_split(
    features_with_sentiment, target, test_size=0.2, random_state=42, shuffle=False
)
X_train_without_sentiment, X_test_without_sentiment, _, _ = train_test_split(
    features_without_sentiment, target, test_size=0.2, random_state=42, shuffle=False
)

# Prepare features and target
merged_data = btc_data_with_sentiment  # Ensure merged_data is defined

features_with_sentiment = merged_data[["Close", "sentiment_score"]]
features_without_sentiment = merged_data[["Close"]]
target = merged_data["Close"].shift(-1).dropna()

features_with_sentiment = features_with_sentiment.iloc[:-1]
features_without_sentiment = features_without_sentiment.iloc[:-1]

X_train_with_sentiment, X_test_with_sentiment, y_train, y_test = train_test_split(
    features_with_sentiment, target, test_size=0.2, random_state=42, shuffle=False
)
X_train_without_sentiment, X_test_without_sentiment, _, _ = train_test_split(
    features_without_sentiment, target, test_size=0.2, random_state=42, shuffle=False
)

print("X_train_with_sentiment shape:", X_train_with_sentiment.shape)
print("X_test_with_sentiment shape:", X_test_with_sentiment.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Random Forest Model
rf_with_sentiment = RandomForestRegressor(random_state=42)
rf_with_sentiment.fit(X_train_with_sentiment, y_train)
y_pred_with_sentiment = rf_with_sentiment.predict(X_test_with_sentiment)

rf_without_sentiment = RandomForestRegressor(random_state=42)
rf_without_sentiment.fit(X_train_without_sentiment, y_train)
y_pred_without_sentiment = rf_without_sentiment.predict(X_test_without_sentiment)

# Metrics Calculation
results_rf = {
    "With Sentiment Analysis": {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_with_sentiment)),
        "R2": r2_score(y_test, y_pred_with_sentiment),
    },
    "Without Sentiment Analysis": {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_without_sentiment)),
        "R2": r2_score(y_test, y_pred_without_sentiment),
    }
}
print("Random Forest Results:", results_rf)

# Scaling data for LSTM
scaler = MinMaxScaler()
scaled_features_with_sentiment = scaler.fit_transform(features_with_sentiment)
scaled_features_without_sentiment = scaler.fit_transform(features_without_sentiment)
scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

lookback = 5

X_lstm_with_sentiment = np.array([scaled_features_with_sentiment[i:i+lookback] for i in range(len(scaled_features_with_sentiment) - lookback)])
X_lstm_without_sentiment = np.array([scaled_features_without_sentiment[i:i+lookback] for i in range(len(scaled_features_without_sentiment) - lookback)])
y_lstm = scaled_target[lookback:]

train_size = int(0.8 * len(X_lstm_with_sentiment))
X_train_with_sentiment_lstm, X_test_with_sentiment_lstm = X_lstm_with_sentiment[:train_size], X_lstm_with_sentiment[train_size:]
X_train_without_sentiment_lstm, X_test_without_sentiment_lstm = X_lstm_without_sentiment[:train_size], X_lstm_without_sentiment[train_size:]
y_train_lstm, y_test_lstm = y_lstm[:train_size], y_lstm[train_size:]

# LSTM Model Definition
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Training LSTM
model_with_sentiment = build_lstm_model(X_train_with_sentiment_lstm.shape[1:])
model_with_sentiment.fit(X_train_with_sentiment_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=1)
y_pred_with_sentiment_lstm = model_with_sentiment.predict(X_test_with_sentiment_lstm)

model_without_sentiment = build_lstm_model(X_train_without_sentiment_lstm.shape[1:])
model_without_sentiment.fit(X_train_without_sentiment_lstm, y_train_lstm, epochs=10, batch_size=16, verbose=1)
y_pred_without_sentiment_lstm = model_without_sentiment.predict(X_test_without_sentiment_lstm)

# Inverse scaling predictions
y_pred_with_sentiment_lstm = scaler.inverse_transform(y_pred_with_sentiment_lstm)
y_pred_without_sentiment_lstm = scaler.inverse_transform(y_pred_without_sentiment_lstm)
y_test_actual = scaler.inverse_transform(y_test_lstm)

# Evaluate LSTM Models
results_lstm = {
    "With Sentiment Analysis": {
        "RMSE": np.sqrt(mean_squared_error(y_test_actual, y_pred_with_sentiment_lstm)),
        "R2": r2_score(y_test_actual, y_pred_with_sentiment_lstm),
    },
    "Without Sentiment Analysis": {
        "RMSE": np.sqrt(mean_squared_error(y_test_actual, y_pred_without_sentiment_lstm)),
        "R2": r2_score(y_test_actual, y_pred_with_sentiment_lstm),
    }
}
print("LSTM Results:", results_lstm)

# 8. --- Plotting ---
import matplotlib.dates as mdates  # Import date formatting module

plt.figure(figsize=(15, 8))  # Increased figure size for better readability

# Correctly align dates with y_test and predictions
test_dates = btc_data['Date'].iloc[btc_data.shape[0] - len(y_test):]

plt.plot(test_dates[lookback:], y_test[lookback:], label="Actual", color="black", linewidth=2)

# Correctly slice predictions to match test_dates[lookback:]
plt.plot(test_dates[lookback:], y_pred_with_sentiment_lstm[:len(test_dates)-lookback], label="LSTM with Sentiment", linestyle="--", color="blue")
plt.plot(test_dates[lookback:], y_pred_without_sentiment_lstm[:len(test_dates)-lookback], label="LSTM without Sentiment", linestyle="--", color="red")

plt.title("Actual vs Predicted Bitcoin Prices", fontsize=16)  # Increased title font size
plt.xlabel("Date", fontsize=14)  # Increased x-axis label font size
plt.ylabel("Price", fontsize=14)  # Increased y-axis label font size
plt.legend(fontsize=12)  # Increased legend font size

# Format x-axis dates for better readability
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())  # Set tick locations to be at the beginning of each month
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # Set the format of the date ticks
plt.gcf().autofmt_xdate()  # Rotate date labels if they overlap

plt.grid(True, linestyle='--', alpha=0.7)  # Added a subtle grid for better visual separation
plt.tight_layout()  # Adjust layout to prevent labels from overlapping

plt.show()

# 9. --- Next Day Prediction ---

# Define and fit the scalers (if not already done earlier in your notebook)
scaler_with_sentiment = MinMaxScaler()
scaled_features_with_sentiment = scaler_with_sentiment.fit_transform(features_with_sentiment)  # Fit and transform training data

scaler = MinMaxScaler()  # Scaler for features without sentiment
scaled_features_without_sentiment = scaler.fit_transform(features_without_sentiment)  # Fit and transform training data

scaler_target = MinMaxScaler()
scaled_target = scaler_target.fit_transform(target.values.reshape(-1, 1))

# Prepare the last 'lookback' days of data
last_data_with_sentiment = features_with_sentiment.iloc[-lookback:].values
last_data_without_sentiment = features_without_sentiment.iloc[-lookback:].values

# Scale the last 'lookback' days of data - USE THE CORRECT SCALER AND RESHAPE
last_data_scaled_with_sentiment = scaler_with_sentiment.transform(last_data_with_sentiment).reshape(1, lookback, features_with_sentiment.shape[1])  # Use scaler_with_sentiment and reshape
last_data_scaled_without_sentiment = scaler.transform(last_data_without_sentiment).reshape(1, lookback, features_without_sentiment.shape[1])  # Use scaler and reshape

# Make predictions
next_day_prediction_with_sentiment = model_with_sentiment.predict(last_data_scaled_with_sentiment)
next_day_prediction_without_sentiment = model_without_sentiment.predict(last_data_scaled_without_sentiment)

# Inverse scale the predictions
next_day_prediction_with_sentiment = scaler_target.inverse_transform(next_day_prediction_with_sentiment)  # Use scaler_target
next_day_prediction_without_sentiment = scaler_target.inverse_transform(next_day_prediction_without_sentiment)  # Use scaler_target

print("Next Day Prediction (with sentiment):", next_day_prediction_with_sentiment[0, 0])
print("Next Day Prediction (without sentiment):", next_day_prediction_without_sentiment[0, 0])

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Calculate regression metrics for LSTM with sentiment analysis
mse_with_sentiment = mean_squared_error(y_test_actual, y_pred_with_sentiment_lstm)
mae_with_sentiment = mean_absolute_error(y_test_actual, y_pred_with_sentiment_lstm)
r2_with_sentiment = r2_score(y_test_actual, y_pred_with_sentiment_lstm)
rmse_with_sentiment = np.sqrt(mse_with_sentiment)

# Calculate regression metrics for LSTM without sentiment analysis
mse_without_sentiment = mean_squared_error(y_test_actual, y_pred_without_sentiment_lstm)
mae_without_sentiment = mean_absolute_error(y_test_actual, y_pred_without_sentiment_lstm)
r2_without_sentiment = r2_score(y_test_actual, y_pred_without_sentiment_lstm)
rmse_without_sentiment = np.sqrt(mse_without_sentiment)

# Print regression metrics for both models
print("Regression Metrics for LSTM with Sentiment Analysis:")
print(f"Mean Squared Error (MSE): {mse_with_sentiment:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_with_sentiment:.4f}")
print(f"Mean Absolute Error (MAE): {mae_with_sentiment:.4f}")
print(f"R-squared (R²): {r2_with_sentiment:.4f}")

print("\nRegression Metrics for LSTM without Sentiment Analysis:")
print(f"Mean Squared Error (MSE): {mse_without_sentiment:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse_without_sentiment:.4f}")
print(f"Mean Absolute Error (MAE): {mae_without_sentiment:.4f}")
print(f"R-squared (R²): {r2_without_sentiment:.4f}")



import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Define a threshold to binarize the predictions
threshold = np.median(y_test_actual)  # Use numpy's median function for arrays
y_test_binary = (y_test_actual > threshold).astype(int)
y_pred_binary = (y_pred_with_sentiment_lstm > threshold).astype(int)

# Calculate classification metrics
accuracy = accuracy_score(y_test_binary, y_pred_binary)
precision = precision_score(y_test_binary, y_pred_binary, zero_division=0)
recall = recall_score(y_test_binary, y_pred_binary, zero_division=0)
f1 = f1_score(y_test_binary, y_pred_binary, zero_division=0)

# Calculate specificity
tn, fp, fn, tp = confusion_matrix(y_test_binary, y_pred_binary).ravel()
specificity = tn / (tn + fp)

print("Classification Metrics for Binarized LSTM Predictions:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score: {f1:.4f}")



from sklearn.linear_model import LinearRegression

# Train individual models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_with_sentiment, y_train)

svr_model = SVR(kernel='rbf')
svr_model.fit(X_train_with_sentiment, y_train)

# LSTM Model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train_with_sentiment.shape[1], 1)))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Reshape data for LSTM
X_train_lstm = X_train_with_sentiment.values.reshape((X_train_with_sentiment.shape[0], X_train_with_sentiment.shape[1], 1))
X_test_lstm = X_test_with_sentiment.values.reshape((X_test_with_sentiment.shape[0], X_test_with_sentiment.shape[1], 1))

# Train LSTM
lstm_model.fit(X_train_lstm, y_train, epochs=10, batch_size=16, verbose=1)

# Get predictions from each model
rf_preds = rf_model.predict(X_test_with_sentiment)
svr_preds = svr_model.predict(X_test_with_sentiment)
lstm_preds = lstm_model.predict(X_test_lstm).flatten()

# Stack predictions as features for meta-model
stacked_predictions = np.column_stack((rf_preds, svr_preds, lstm_preds))

# Train meta-model
meta_model = LinearRegression()
meta_model.fit(stacked_predictions, y_test)

# Final ensemble predictions
final_preds = meta_model.predict(stacked_predictions)

# Evaluate model performance
ensemble_rmse = mean_squared_error(y_test, final_preds, squared=False)
ensemble_r2 = r2_score(y_test, final_preds)

print(f"Ensemble RMSE: {ensemble_rmse}")
print(f"Ensemble R2 Score: {ensemble_r2}")




