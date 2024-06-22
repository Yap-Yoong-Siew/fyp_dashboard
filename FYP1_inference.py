# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 15:01:10 2024

@author: user
"""

import pandas as pd
from math import sin, cos, pi
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from xgboost import XGBClassifier
import xgboost as xgb
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
import joblib
import psycopg2
import psycopg2.extensions
import select
import json
import numpy as np
# Load the saved model
rf_model = joblib.load('random_forest_model.joblib')

db_params = {
    'dbname': 'postgres',  # Your database name
    'user': 'postgres',    # Your database username
    'password': '1234',    # Your database password
    'host': 'localhost',   # Host where the database server is running
    'port': '5432'         # Port where the database server is listening
}

# Connect to the PostgreSQL database
conn = psycopg2.connect(dbname="postgres", user="postgres", password="1234", host="localhost")
conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)# Create a cursor
cur = conn.cursor()

# Listen for notifications on the 'my_table_notification' channel
cur.execute("LISTEN new_row_alert;")
print("Waiting for notifications on channel 'new_row_alert'")

# Function to handle received notifications
def handle_notification(conn, notification):
    print(f"Received notification: {notification.payload}")
    # Process the notification payload as needed

def create_table():
    with psycopg2.connect(**db_params) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS trade (
                      hour INT,
                      volume INT,
                      atr NUMERIC(10, 5),
                      timestamp timestamp with time zone,
                      allow INT,
                      PRIMARY KEY (hour, volume, atr, allow, timestamp)
                )
            """)
            conn.commit()
create_table()
while True:
    if select.select([conn], [], [], 5) == ([], [], []):
        print("Listening")
    else:
        conn.poll()
        while conn.notifies:
            notify = conn.notifies.pop(0)
            print("Got NOTIFY:", notify.pid, notify.channel, notify.payload)
            payload_str = notify.payload
            payload_dict = json.loads(payload_str)  # Convert JSON string to dictionary
            hour = payload_dict['hour']
            volume = payload_dict['volume']
            atr = payload_dict['atr']
            hour_sin = sin(2 * pi * hour / 23)
            hour_cos = cos(2 * pi * hour / 23)
            data = {'hour_sin' : [hour_sin],
                    'hour_cos' : [hour_cos],
                    'tick_volume' : [volume],
                    'atr': [atr]}
            X_test = pd.DataFrame(data)
            print(f"X-test is {X_test}")
            y_pred = rf_model.predict(X_test)
            print(f"y_pred is {y_pred}")
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO trade (hour, volume, atr, allow, timestamp) 
                    VALUES (%s, %s, %s, %s, %s)
                """, (hour, volume, atr, int(y_pred[0]), payload_dict['timestamp']))
                conn.commit()
            
            
            
            