#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import os
from database import save_to_database
from ml_model import train_lstm_model, predict_lstm
from utils import load_data

st.set_page_config(page_title="Walmart Sales Forecasting")

st.title("📊 Walmart Store Sales Forecasting")
st.markdown("Upload your datasets and train an LSTM model to forecast sales.")

# Upload files
train_file = st.file_uploader("Upload Train.csv", type="csv")
features_file = st.file_uploader("Upload Features.csv", type="csv")
stores_file = st.file_uploader("Upload Stores.csv", type="csv")

if train_file and features_file and stores_file:
    train_df = pd.read_csv(train_file)
    features_df = pd.read_csv(features_file)
    stores_df = pd.read_csv(stores_file)

    # Save locally
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    features_df.to_csv("data/features.csv", index=False)
    stores_df.to_csv("data/stores.csv", index=False)

    # Store to DB
    save_to_database(train_df, features_df, stores_df)

    st.success("✅ Files uploaded and stored in database.")

    if st.button("Preview Train Data"):
        st.dataframe(train_df.head())

    st.subheader("Train Model")
    store_id = st.number_input("Store ID", min_value=1, max_value=45)
    dept_id = st.number_input("Department ID", min_value=1, max_value=99)

    if st.button("Train LSTM Model"):
        rmse = train_lstm_model(store_id, dept_id)
        st.success(f"Model trained! RMSE: {rmse:.2f}")

    if st.button("Predict Future Sales"):
        result_df = predict_lstm(store_id, dept_id)
        st.line_chart(result_df.set_index("Date"))

