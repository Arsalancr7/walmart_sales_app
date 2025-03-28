#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from database import get_merged_data

def train_lstm_model(store_id, dept_id):
    df = get_merged_data()
    df = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)].sort_values("Date")
    df.fillna(0, inplace=True)

    features = ["Weekly_Sales", "Temperature", "Fuel_Price", "MarkDown1", "CPI", "Unemployment"]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[features])
    
    X, y = [], []
    n_steps = 10
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps][0])
    X, y = np.array(X), np.array(y)

    split = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=20, verbose=0)

    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    model.save(f"models/store{store_id}_dept{dept_id}.h5")

    return rmse

def predict_lstm(store_id, dept_id):
    from tensorflow.keras.models import load_model

    df = get_merged_data()
    df = df[(df["Store"] == store_id) & (df["Dept"] == dept_id)].sort_values("Date")
    df.fillna(0, inplace=True)
    features = ["Weekly_Sales", "Temperature", "Fuel_Price", "MarkDown1", "CPI", "Unemployment"]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(df[features])
    
    n_steps = 10
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:i+n_steps])
        y.append(data[i+n_steps][0])
    X = np.array(X)
    
    model = load_model(f"models/store{store_id}_dept{dept_id}.h5")
    preds = model.predict(X)

    result_df = pd.DataFrame({
        "Date": df["Date"].values[n_steps:],
        "True_Sales": [x[0] for x in data[n_steps:]],
        "Predicted_Sales": preds.flatten()
    })

    return result_df

