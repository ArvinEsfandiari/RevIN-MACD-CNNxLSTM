import MetaTrader5 as mt5
import pandas as pd
import os
import datetime

def initialize_mt5():
    """Initialize the MT5 connection"""
    if not mt5.initialize():
        print("MetaTrader 5 initialization failed")
        return False
    print("MetaTrader 5 initialized")
    return True

def fetch_data_from_mt5(symbol, timeframe, start_date, end_date):
    """
    Fetch historical data from MetaTrader 5.

    :param symbol: The symbol to fetch data for (e.g., 'BTCUSD')
    :param timeframe: The time frame to use (e.g., mt5.TIMEFRAME_M1 for 1-minute data)
    :param start_date: The start date for fetching data
    :param end_date: The end date for fetching data
    :return: Pandas DataFrame with historical data
    """
    initialize_mt5()

    # Get the symbol info
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        print(f"{symbol} not found")
        return None

    # Get historical data
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        print("No data fetched")
        return None

    # Create DataFrame
    data = pd.DataFrame(rates)
    data['time'] = pd.to_datetime(data['time'], unit='s')
    data = data.set_index('time')

    return data


def save_data_to_csv(data, symbol, timeframe):
    """
    Save the fetched data to a CSV file.

    :param data: The DataFrame containing the market data
    :param symbol: The symbol of the data (e.g., 'BTCUSD')
    :param timeframe: The time frame of the data
    """

    # Get the absolute path of the directory where this script is located
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Ensure the 'data' directory exists within the base directory
    #data_dir = base_dir.replace('src','data')
    data_dir = os.path.join(base_dir, 'data')  
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        print(f"Created 'data' directory at {data_dir}")

    today = datetime.datetime.today().strftime('%Y-%m-%d')
    # Save the file in the 'data' folder
    filename = f"{symbol}_{timeframe}_{today}.csv"
    filepath = os.path.join(data_dir, filename)
    
    # Save DataFrame to CSV
    data.to_csv(filepath)
    print(f"Data saved to {filepath}")


def data_cleaner(data):


    


    return



symbol = 'BTCUSD'
timeframe = mt5.TIMEFRAME_M5  # 1-minute time frame
start_date = datetime.datetime(2023, 11, 30)
end_date = datetime.datetime(2024, 10, 30)


# Fetch data
data = fetch_data_from_mt5(symbol, timeframe, start_date, end_date)

# Save to CSV
save_data_to_csv(data, symbol, "5m")