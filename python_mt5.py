# -*- coding: utf-8 -*-
"""
Created on Fri May 26 07:53:54 2023

@author: user
"""

import MetaTrader5
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
#%%

username = 51823099
password = "9XPNgq@6GyxtVg"
server = "ICMarketsSC-Demo"
path = "C:\\Program Files\\MetaTrader 5 IC Markets(SC)\\terminal64.exe"

# Function to start Meta Trader 5 (MT5)
def start_mt5(username, password, server, path):
    # Ensure that all variables are the correct type
    uname = int(username) # Username must be an int
    pword = str(password) # Password must be a string
    trading_server = str(server) # Server must be a string
    filepath = str(path) # Filepath must be a string

    # Attempt to start MT5
    if MetaTrader5.initialize(login=uname, password=pword, server=trading_server, path=filepath):
        # print("Trading Bot Starting")
        pass
        # Login to MT5
        if MetaTrader5.login(login=uname, password=pword, server=trading_server):
            # print("Trading Bot Logged in and Ready to Go!")
            return True
        else:
            print("Login Fail")
            quit()
            return PermissionError
    else:
        print("MT5 Initialization Failed")
        quit()
        return ConnectionAbortedError
    
    
def initialize_symbols(symbol_array):
    # Get a list of all symbols supported in MT5
    all_symbols = MetaTrader5.symbols_get()
    # Create an array to store all the symbols
    symbol_names = []
    # Add the retrieved symbols to the array
    for symbol in all_symbols:
        symbol_names.append(symbol.name)

    # Check each symbol in symbol_array to ensure it exists
    for provided_symbol in symbol_array:
        if provided_symbol in symbol_names:
            # If it exists, enable
            if MetaTrader5.symbol_select(provided_symbol, True):
                # print(f"Sybmol {provided_symbol} enabled")
                pass
            else:
                return ValueError
        else:
            return SyntaxError

    # Return true when all symbols enabled
    return True




# Function to query previous candlestick data from MT5
def query_historic_data(symbol, timeframe, number_of_candles):
    # Convert the timeframe into an MT5 friendly format
    mt5_timeframe = set_query_timeframe(timeframe)
    # Retrieve data from MT5
    rates = MetaTrader5.copy_rates_from_pos(symbol, mt5_timeframe, 1, number_of_candles)
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.rename(columns={'time': 'datetime'}, inplace=True)
    df.set_index('datetime', inplace=True)
    df.rename(columns={'open' : 'Open',
                       'high' : 'High',
                       'low': 'Low',
                       'close': 'Close'}, inplace=True)
    return df


# Function to convert a timeframe string in MetaTrader 5 friendly format
def set_query_timeframe(timeframe):
    # Implement a Pseudo Switch statement. Note that Python 3.10 implements match / case but have kept it this way for
    # backwards integration
    if timeframe == "M1":
        return MetaTrader5.TIMEFRAME_M1
    elif timeframe == "M2":
        return MetaTrader5.TIMEFRAME_M2
    elif timeframe == "M3":
        return MetaTrader5.TIMEFRAME_M3
    elif timeframe == "M4":
        return MetaTrader5.TIMEFRAME_M4
    elif timeframe == "M5":
        return MetaTrader5.TIMEFRAME_M5
    elif timeframe == "M6":
        return MetaTrader5.TIMEFRAME_M6
    elif timeframe == "M10":
        return MetaTrader5.TIMEFRAME_M10
    elif timeframe == "M12":
        return MetaTrader5.TIMEFRAME_M12
    elif timeframe == "M15":
        return MetaTrader5.TIMEFRAME_M15
    elif timeframe == "M20":
        return MetaTrader5.TIMEFRAME_M20
    elif timeframe == "M30":
        return MetaTrader5.TIMEFRAME_M30
    elif timeframe == "H1":
        return MetaTrader5.TIMEFRAME_H1
    elif timeframe == "H2":
        return MetaTrader5.TIMEFRAME_H2
    elif timeframe == "H3":
        return MetaTrader5.TIMEFRAME_H3
    elif timeframe == "H4":
        return MetaTrader5.TIMEFRAME_H4
    elif timeframe == "H6":
        return MetaTrader5.TIMEFRAME_H6
    elif timeframe == "H8":
        return MetaTrader5.TIMEFRAME_H8
    elif timeframe == "H12":
        return MetaTrader5.TIMEFRAME_H12
    elif timeframe == "D1":
        return MetaTrader5.TIMEFRAME_D1
    elif timeframe == "W1":
        return MetaTrader5.TIMEFRAME_W1
    elif timeframe == "MN1":
        return MetaTrader5.TIMEFRAME_MN1
    
#%%
    
    
start_mt5(username, password, server, path)
initialize_symbols(["EURUSD", "GBPUSD", "USDJPY", "AUDNZD", "EURNZD","EURAUD","AUDNZD","NZDCAD", "AUDCAD"])

ticker = "EURUSD"
rates = query_historic_data(ticker, "M5", 2000)
#%%
# df_prices = rates
# df_prices['price_range'] = (df_prices['high'] - df_prices['low']) * 10000
# #%%
# # Create a plot
# # plt.figure(figsize=(14, 7))
# # plt.plot(df_prices['time'], df_prices['price_range'], 'o')

# # plt.title(f'Daily Price Range {ticker}')
# # plt.xlabel('Date')
# # plt.ylabel('Price Range in pips')
# # plt.grid(True)
# # plt.show()

# #%%


# df_news = pd.read_csv('NewsEvents.txt')
# df_news = df_news.drop('Speech?', axis=1)
# df_news['DateTime'] = pd.to_datetime(df_news['DateTime']).dt.date
# df_news = df_news[df_news['Impact'] == 'High']

# #%%
# df_news_pivot = df_news.pivot_table(index='DateTime', columns='Pair', values='Impact', aggfunc='any')
# df_news_pivot = df_news_pivot.notnull().astype('int')
# df_prices['time'] = pd.to_datetime(df_prices['time']).dt.date
# df_news = df_news[df_news['Pair'].isin(['USD', 'AUD', 'NZD'])]

# df = df_prices.join(df_news_pivot, on='time')


# df['color'] = df.apply(lambda row: 'AUD' if row['AUD']==1 else('NZD' if row['NZD'] else ('USD' if row['USD']==1 else 'No News')), axis=1)



# #%%

# def plot_news_impact_on_price_range(ticker, priorities, num_days):

#     rates = query_historic_data(ticker, "D1", num_days)
#     df_prices = rates
#     df_prices['price_range'] = (df_prices['high'] - df_prices['low']) * 10000
    
#     df_news = pd.read_csv('NewsEvents.txt')
#     df_news = df_news.drop('Speech?', axis=1)
#     df_news['DateTime'] = pd.to_datetime(df_news['DateTime']).dt.date
#     df_news = df_news[df_news['Impact'] == 'High']
    
#     df_news_pivot = df_news.pivot_table(index='DateTime', columns='Pair', values='Impact', aggfunc='any')
#     df_news_pivot = df_news_pivot.notnull().astype('int')
#     df_prices['time'] = pd.to_datetime(df_prices['time']).dt.date
#     df_news = df_news[df_news['Pair'].isin(priorities)]
    
#     df = df_prices.join(df_news_pivot, on='time')
#     df.fillna(0, inplace=True)  # fill NA values with 0
#     # df['color'] = df[priorities].apply(lambda row: next((priority for priority, value in zip(priorities, row) if value==1), 'No News'), axis=1)
    
#     df['color'] = df[priorities].apply(
#         lambda row: 'No News' if all(value == 0 for value in row) else next((priority for priority, value in zip(priorities, row) if value==1), 'No News'), axis=1)
    
    
#     # df['color'] = df[priorities].idxmax(axis=1)
    
#     fig, ax = plt.subplots(figsize=(14, 7))
    
#     colors = {**{priority: plt.get_cmap('tab10')(i) for i, priority in enumerate(priorities)}, 'No News': 'red'}
#     color_values = df['color'].map(colors).values
    
#     ax.scatter(df['time'], df['price_range'], color=color_values)
    
#     # Create a custom legend
#     from matplotlib.lines import Line2D
#     color_counts = df['color'].value_counts()
#     avg_pips = df.groupby('color')['price_range'].mean()
#     custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors.values()]
    
#     legend_labels = [f'{key} - {color_counts[key]} dots - {avg_pips[key]:.2f} pips' for key in colors.keys()]
#     ax.legend(custom_lines, legend_labels, title="News Origin")
    
#     ax.set_xlabel('Date')
#     ax.set_ylabel('Price change in pips')
#     ax.set_title(f'Daily Price Range {ticker}')
    
#     # Format the x axis
#     date_form = DateFormatter("%Y-%m-%d")
#     ax.xaxis.set_major_formatter(date_form)
#     ax.grid(True)
#     plt.show()
# #%%
# # use the function
# if __name__ == "__main__":
#     priorities = ['CHF', 'AUD', 'USD'] 
#     ticker = "AUDCHF"
#     plot_news_impact_on_price_range(ticker, priorities, 1500)
















