import pandas as pd
import numpy as np

class ImprovedBacktest:
    def __init__(self, data, fast_period, slow_period, signal_period, price_type='close',
                 sell_fee=0.115, buy_fee=0.115, initial_capital=100000.0):
        self.data = data.copy()
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_type = price_type
        self.sell_fee_percent = sell_fee / 100
        self.buy_fee_percent = buy_fee / 100
        self.initial_capital = initial_capital
        self.trades = []
        self.data['position'] = 0

    def calculate_macd(self):
        """
        Calculates the MACD indicator.
        """
        # Compute the EMAs
        self.data['ema_fast'] = self.data[self.price_type].ewm(span=self.fast_period, adjust=False).mean()
        self.data['ema_slow'] = self.data[self.price_type].ewm(span=self.slow_period, adjust=False).mean()
        # Calculate MACD components
        self.data['macd_line'] = self.data['ema_fast'] - self.data['ema_slow']
        self.data['signal_line'] = self.data['macd_line'].ewm(span=self.signal_period, adjust=False).mean()

    def generate_signals(self):
        """
        Generates trading signals based on MACD crossover.
        """
        self.data['signal'] = 0
        self.data['signal'][self.signal_period:] = np.where(
            self.data['macd_line'][self.signal_period:] > self.data['signal_line'][self.signal_period:], 1, 0)
        # Generate trading positions by taking the difference of the signals
        self.data['position'] = self.data['signal'].shift(1).fillna(0)

    def backtest_strategy(self):
        """
        Backtests the trading strategy.
        """
        self.calculate_macd()
        self.generate_signals()

        data = self.data.copy()
        data['price'] = data[self.price_type]

        cash = self.initial_capital
        holdings = 0
        portfolio_values = []
        positions = []

        for i in range(len(data)):
            price = data['price'].iloc[i]
            position = data['position'].iloc[i]
            date = data.index[i]

            # Check for buy signal
            if position > holdings:
                units_to_buy = position - holdings
                buy_price = price
                total_cost = units_to_buy * buy_price * (1 + self.buy_fee_percent)
                if cash >= total_cost:
                    cash -= total_cost
                    holdings += units_to_buy
                    self.trades.append({
                        'type': 'buy',
                        'units': units_to_buy,
                        'price': buy_price,
                        'fee': buy_price * units_to_buy * self.buy_fee_percent,
                        'date': date
                    })

            # Check for sell signal
            elif position < holdings:
                units_to_sell = holdings - position
                sell_price = price
                total_proceeds = units_to_sell * sell_price * (1 - self.sell_fee_percent)
                cash += total_proceeds
                holdings -= units_to_sell
                self.trades.append({
                    'type': 'sell',
                    'units': units_to_sell,
                    'price': sell_price,
                    'fee': sell_price * units_to_sell * self.sell_fee_percent,
                    'date': date
                })

            # Calculate total portfolio value
            total_value = cash + holdings * price
            portfolio_values.append(total_value)
            positions.append(holdings)

        self.data['portfolio_value'] = portfolio_values
        self.data['holdings'] = positions
        total_return = ((self.data['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital) * 100
        return total_return

    def print_trade_summary(self):
        """
        Prints a summary of all trades.
        """
        total_return_percent = 0
        for i in range(len(self.trades)):
            trade = self.trades[i]
            if trade['type'] == 'buy':
                try:
                    sell_trade = next(
                        t for t in self.trades[i+1:] if t['type'] == 'sell' and t['units'] == trade['units']
                    )
                    buy_cost = trade['units'] * trade['price'] + trade['fee']
                    sell_proceeds = sell_trade['units'] * sell_trade['price'] - sell_trade['fee']
                    trade_return = ((sell_proceeds - buy_cost) / buy_cost) * 100
                    total_return_percent += trade_return
                    print(f"Trade {i//2 +1}: Buy {trade['units']} units at {trade['price']:.2f} on {trade['date']}, "
                          f"Sell at {sell_trade['price']:.2f} on {sell_trade['date']}, Return: {trade_return:.2f}%")
                except StopIteration:
                    continue  # No corresponding sell trade found yet
        print(f"Total Return: {total_return_percent:.2f}%")

    def get_total_return(self):
        """
        Returns the total return percentage.
        """
        return (self.data['portfolio_value'].iloc[-1] - self.initial_capital) / self.initial_capital * 100

    def get_portfolio(self):
        """
        Returns the portfolio value over time.
        """
        return self.data[['portfolio_value', 'holdings']]

