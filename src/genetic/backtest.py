import pandas as pd
import numpy as np

class backtest:
    def __init__(self, data, fast_period, slow_period, signal_period, price_type = 'close'):
        
        self.data = data
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_type = price_type

    def calculate_macd(self):
        """
        Calculates the MACD indicator.

        Parameters:
        - data: The DataFrame containing the price data.
        - fast_period: The period for the fast EMA.
        - slow_period: The period for the slow EMA.
        - signal_period: The period for the signal line EMA.
        """
        data = pd.DataFrame()
        data['time'] = self.data.index
        data.set_index('time', inplace=True)
        data[self.price_type] = self.data[self.price_type]
        # Compute the EMAs
        ema_fast = data[self.price_type].ewm(span=self.fast_period, adjust=False).mean()
        ema_slow = data[self.price_type].ewm(span=self.slow_period, adjust=False).mean()
        # Calculate MACD components
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal_period, adjust=False).mean()
        #histogram = macd_line - signal_line
        # Add to DataFrame
        data['macd_line'] = macd_line
        data['signal_line'] = signal_line
        #data['histogram'] = histogram
        return data
    
    def generate_signals(self, data):
        """
        Generates trading signals based on MACD crossover.

        Parameters:
        - data: The DataFrame containing the MACD data.
        """
        # Corrected assignment to avoid SettingWithCopyWarning
        data.loc[data.index[1:], 'signal'] = np.where(
            data['macd_line'].iloc[1:] > data['signal_line'].iloc[1:], 1, 0)
        data['positions'] = data['signal'].diff()
        
        return data


    def backtest_strategy(self, data):
        """
        Backtests the trading strategy.

        Parameters:
        - data: The DataFrame containing the signals.
        """
        initial_capital = float(100000.0)
        positions = pd.DataFrame(index=data.index).fillna(0.0)
        positions['positions'] = data['signal'].shift(1).fillna(0)
        # Calculate holdings and cash
        positions['holdings'] = positions['positions'] * data[self.price_type]
        positions['cash'] = initial_capital - (
            (positions['positions'].diff() * data[self.price_type]).fillna(0).cumsum()
        )
        positions['total'] = positions['cash'] + positions['holdings']
        positions['returns'] = positions['total'].pct_change().fillna(0)
        total_return = positions['total'].iloc[-1] - initial_capital
        return total_return
    

    