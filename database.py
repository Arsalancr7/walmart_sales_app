#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sqlite3

def save_to_database(train_df, features_df, stores_df):
    conn = sqlite3.connect("walmart.db")
    train_df.to_sql("train", conn, if_exists="replace", index=False)
    features_df.to_sql("features", conn, if_exists="replace", index=False)
    stores_df.to_sql("stores", conn, if_exists="replace", index=False)
    conn.close()

def get_merged_data():
    conn = sqlite3.connect("walmart.db")
    query = """
        SELECT 
            t.Store, t.Dept, t.Date, t.Weekly_Sales, t.IsHoliday,
            f.Temperature, f.Fuel_Price, f.MarkDown1, f.MarkDown2, f.MarkDown3,
            f.CPI, f.Unemployment, s.Type, s.Size
        FROM train t
        JOIN features f ON t.Store = f.Store AND t.Date = f.Date
        JOIN stores s ON t.Store = s.Store
        ORDER BY t.Store, t.Dept, t.Date;
    """
    df = pd.read_sql(query, conn, parse_dates=["Date"])
    conn.close()
    return df

