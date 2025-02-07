import numpy as np
import pandas as pd

class MACDBacktester:
    def __init__(self, data, signal_price = 'smoothed_data', real_price = 'close',
                  sell_fee = 0.115, buy_fee = 0.115, initial_capital = 100):
        """"
        It works as a trade andlyzer with an specific amount of money and trading cost.

        :param data: Dataframe contains different features
        :param fast_ema: Integer
        :param slow_ema: Ineger
        :param signal_line: Integer
        :param signal_price: for producing buy and sell signals(defualt:'close')
        :param real_price: for trading(defualt:'close')
        :param sell_fee: Float contains sell trading cost(defulat=0.115). Nobitex maker trading cost is 0.1 and taker is 0.13.
        :param buy_fee: Float contains buy trading cost(defulat=0.115). Nobitex
        :param initial_capital: Integer contains initial money to start.

        """
        self.data = data.copy()
        self.signal_price = signal_price
        self.real_price = real_price
        self.sell_fee_percent = sell_fee / 100
        self.buy_fee_percent = buy_fee / 100
        self.initial_capital = initial_capital
        self.trades = []
        self.data['positions'] = 0
        
    # def calculate_macd(self):
    #     self.data['fast_ema'] = self.data[self.signal_price].ewm(span = self.fast_ema, adjust = False).mean()
    #     self.data['slow_ema'] = self.data[self.signal_price].ewm(span = self.slow_ema, adjust = False).mean()
    #     self.data['macd_line'] = self.data['fast_ema'] - self.data['slow_ema']
    #     self.data['signal_line'] = self.data['macd_line'].ewm(span = self.signal_line, adjust = False).mean()


    def generate_signals(self):
        """
        Generates trading signals based on MACD crossover.
        """
        threshold = 0.1
        self.data['signal'] = 0
        self.data['signal'] = np.where((self.data['macd_line'] - self.data['signal_line']) > threshold, 1, 0) # Buy signal
        self.data['signal'] = np.where((self.data['macd_line'] - self.data['signal_line']) < -threshold, 0, self.data['signal']) # Sell signal
        # Generation the position by shifting the signals
        #self.data['positions'] = self.data['signal'].shift(1).fillna(0)
        self.data['positions'] = self.data['signal']
        return self.data

    def backtest_strategy(self):
        """
        Backtests the strategy and calculates performance metrics.
        """
        self.data['price'] = self.data[self.real_price]
        self.data['positions'] = self.data['positions'].astype(int)
        self.data['positions_diff'] = self.data['positions'].diff()
        self.data['positions_diff'].fillna(0)

        # Initialize cash and holdings
        self.data['cash'] = self.initial_capital
        self.data['holdings'] = 0.0
        self.data['total'] = self.initial_capital


        # Variable to keep track of cash, holdings, trades
        cash = self.initial_capital
        holdings = 0.0
        position = 0  # Current position (number of shares)
        buy_price = 0.0
        win_count = 0  # Win rate calculation
        total_trades = 0

        for idx, row in self.data.iterrows():
            position_change = row['positions_diff']
            price = row['price']
            if position_change == 1: # Long position
                # With regard to the randomness of market, I can decide how much of cash should spend for trading.
                shares_to_buy = round(cash/(price),8)
                shares_to_buy = shares_to_buy * (1-self.buy_fee_percent)

                if shares_to_buy > 0.00002:
                    buy_price = round((cash/shares_to_buy),8) 
                    total_cost = round(shares_to_buy * buy_price, 8)
                    cash -= total_cost
                    holdings += shares_to_buy * price
                    position += shares_to_buy


            elif position_change ==-1 and position > 0: # position is the number of shares.
                # Exit the long position
                sell_price = price * (1-self.sell_fee_percent)
                total_proceeds = position * sell_price
                cash += total_proceeds
                holdings -= position * sell_price
                # Calculate trade return
                trade_return = (sell_price - buy_price)/buy_price * 100
                self.trades.append(trade_return)
                position = 0
                total_trades +=1 # Win rate calculation
                if trade_return>0:
                    win_count +=1
            else:
                # Hold position
                holdings = position * price


            total = cash + holdings
            self.data.at[idx, 'cash'] = float(cash)
            self.data.at[idx, 'holdings'] = holdings
            self.data.at[idx, 'total'] = float(total)

        if position > 0 :
            
            price = self.data.iloc[-1]['price']
            sell_price = price * (1-self.sell_fee_percent)
            total_proceeds = position * sell_price
            cash += total_proceeds
            # Calculate trade return
            trade_return = (sell_price - buy_price) / buy_price * 100
            self.trades.append(trade_return)
            position = 0
            total_trades +=1 # Win rate calculation
            if trade_return>0:
                win_count +=1
            total = cash + holdings
            self.data.at[self.data.index[-1], 'cash'] = cash
            self.data.at[self.data.index[-1], 'holdings'] = holdings
            self.data.at[self.data.index[-1], 'total'] = total

        self.results = self.data[['cash', 'holdings', 'total']]
        self.results = self.data[['cash', 'holdings', 'total']]

        self.win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        return self.data

    def get_performance_metrics(self):
        """
        Calculates and returns performance metrics.
        """
        if self.results is None:
            print("Please run backtest_strategy() before calculating performance metrics.")
            return None

        total_return = (self.results['total'].iloc[-1] - self.initial_capital) / self.initial_capital * 100
        returns = self.results['total'].pct_change().fillna(0)
        annualized_return = ((1 + returns.mean()) ** 365 - 1) * 100  # Assuming daily returns
        annualized_volatility = returns.std() * np.sqrt(365) * 100

          # Calculating sharpe ration
        periods_per_year = 8760

        # USA risk-free rate assumptions:
        # Assume an annual USA risk-free rate of 4%
        annual_rf_rate = 0.04  
        # Convert to a daily risk-free rate (using 365 days)
        daily_rf_rate = annual_rf_rate / 365  
        # Convert daily rate to an hourly rate (24 hours per day)
        hourly_rf_rate = daily_rf_rate / 24  

        # Calculate the excess hourly returns (return minus hourly risk-free rate)
        excess_returns = returns - hourly_rf_rate

        # Compute the mean and standard deviation of the hourly excess returns
        mean_hourly_excess = excess_returns.mean()
        std_hourly = returns.std()  # Using raw returns' std; you could also use excess_returns.std() if preferred

        # Annualize the Sharpe Ratio (even with less than one year of data)
        sharpe_ratio = (mean_hourly_excess / std_hourly) * np.sqrt(periods_per_year) if std_hourly != 0 else np.nan

        # Caculate Sortino Ratio
        target_return = hourly_rf_rate  
        downside_returns = returns.copy()
        downside_returns[downside_returns > target_return] = 0
        std_downside = downside_returns.std()
        annualized_downside_deviation = std_downside * np.sqrt(periods_per_year)

        # Annualized excess return (arithmetic) for Sortino ratio
        annualized_excess_return = mean_hourly_excess * periods_per_year

        # Sortino Ratio: excess return divided by annualized downside deviation
        sortino_ratio = annualized_excess_return / annualized_downside_deviation if annualized_downside_deviation != 0 else np.nan


        # sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(365) if returns.std() != 0 else np.nan
        max_drawdown = ((self.results['total'].cummax() - self.results['total']) / self.results['total'].cummax()).max() * 100

        metrics = {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Annualized Volatility (%)': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)' : self.win_rate
        }
        return metrics, self.results

    def print_trades(self):
        """
        Prints individual trade returns.
        """
        if not self.trades:
            print("No trades have been executed.")
            return
        for idx, trade_return in enumerate(self.trades, 1):
            print(f"Trade {idx}: Return = {trade_return:.2f}%")
        total_return = sum(self.trades)
        print(f"Total Return from trades: {total_return:.2f}%")