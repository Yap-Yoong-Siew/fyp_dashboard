from flask import Flask, render_template, send_file
import psycopg2
import matplotlib.pyplot as plt
from io import BytesIO
import os

app = Flask(__name__)

# Database credentials - replace with your actual credentials
DATABASE = 'postgres'
USER = 'postgres'
PASSWORD = '1234'
HOST = 'localhost'
PORT = '5432'

def get_db_connection():
    conn = psycopg2.connect(database=DATABASE, user=USER, password=PASSWORD, host=HOST, port=PORT)
    return conn

def get_latest_balance_equity():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT balance, equity, timestamp FROM "Equity_balance" ORDER BY timestamp DESC LIMIT 1;')
    latest_values = cur.fetchone()
    cur.close()
    conn.close()
    return latest_values

def plot_combined_time_series():
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT balance, equity, timestamp FROM "Equity_balance" ORDER BY timestamp;')
        data = cur.fetchall()
        cur.close()
        conn.close()

        times = [row[2] for row in data]
        balances = [row[0] for row in data]
        equities = [row[1] for row in data]
        
        plt.figure(figsize=(10,5))
        plt.plot(times, balances, label='Balance')
        plt.plot(times, equities, label='Equity', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title('Balance and Equity Time Series')
        plt.legend()
        
        img = BytesIO()
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)
        return img

@app.route('/')
def index():
    balance, equity, _ = get_latest_balance_equity()
    return render_template('index.html', balance=balance, equity=equity)

@app.route('/combined_plot.png')
def combined_plot():
    img = plot_combined_time_series()
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
