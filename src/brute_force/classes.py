import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class MACD_calculator:
    def __init__(self, df, price_column):
        self.df = df
        self.price_column = price_column
        self.macd = None
        self.signal_line = None
        self.position = None
        
    def traditional_MACD(self):
        """
        Calculate the traditional MACD and signal line without storing intermediate EMAs
        """
        # Use single-pass operations where possible
        price_df = self.df[self.price_column]
        
        # Calculate EMAs and MACD in a single step, optimizing memory usage
        ema_12 = price_df.ewm(span=12, adjust=False).mean()
        ema_26 = price_df.ewm(span=26, adjust=False).mean()
        self.macd = ema_12 - ema_26
        
        # Calculate the signal line directly from the MACD
        self.signal_line = self.macd.ewm(span=9, adjust=False).mean()
        
        return self  # Return self to allow method chaining

    def optimize_MACD(self, coff_ema_short, coff_ema_long, coff_signal_line):
        """
        Optimize MACD calculation with custom coefficients for short, long EMAs and the signal line.
        """
        price_df = self.df[self.price_column]
        
        # Calculate optimized EMAs with custom coefficients
        ema_short = price_df.ewm(span=round(coff_ema_short), adjust=False).mean()
        ema_long = price_df.ewm(span=round(coff_ema_long), adjust=False).mean()
        self.macd = ema_short - ema_long
        
        # Calculate the signal line with the custom coefficient
        self.signal_line = self.macd.ewm(span=round(coff_signal_line), adjust=False).mean()
        
        return self  # Return self for chaining
    
    def generate_signals(self):
        """
        Generate buy (1) and sell (0) signals based on the MACD crossover with the signal line.
        """
        if self.macd is None or self.signal_line is None:
            raise ValueError("MACD and Signal Line must be calculated before generating signals.")
        
        # Efficient vectorized signal generation using numpy
        signal = np.where(self.macd > self.signal_line, 1, 0)  # 1: Buy, 0: Sell
        self.position = np.diff(signal, prepend=0)  # Calculate positions (buy/sell transitions)
        
        return self  # Return self for chaining
    
    def number_of_trades(self):
        """
        Count the number of buy and sell trades based on the generated signals.
        """
        if self.position is None:
            raise ValueError("Signals must be generated before counting trades.")
        
        # Efficient count of buy and sell trades using numpy's boolean indexing
        buy_count = np.sum(self.position == 1)  # Buy trades (where position == 1)
        sell_count = np.sum(self.position == -1)  # Sell trades (where position == -1)
        
        print(f"The number of buy signals is: {buy_count}")
        print(f"The number of sell signals is: {sell_count}")
        
        return self  # Return self for chaining
    
# Outputs
# .macd and .signal_line for traditional macd and optimize macd
# .position for generate signals
# number of trade with produce automatically



class TradeAnalyzer:
    def __init__(self, df, positions, price_column, sell_fee=0.115, buy_fee = 0.115):
        self.df = df.copy()
        self.df["Positions"] = positions  # Attach the positions to the df
        self.price_column = price_column  # Store the price column name
        self.sell_fee_percent = sell_fee / 100  # Convert fee to decimal
        self.buy_fee_percent = buy_fee / 100
        self.returns = []

    def calculate_return(self):
        # Create buy/sell columns using np.where based on positions
        buy = pd.Series(np.where(self.df["Positions"] == 1, self.df[self.price_column], np.nan))
        sell = pd.Series(np.where(self.df["Positions"] == -1, self.df[self.price_column], np.nan))
        
        # Forward-fill the buy price to match the next sell signal
        buy = buy.ffill()  # Ensure buys are forward-filled until the next sell
        
        # Create valid trades dfFrame with non-null buy-sell pairs
        trades = pd.DataFrame({'Buy': buy, 'Sell': sell}).dropna(subset=['Sell'])

        if trades.empty:
            print("No complete buy-sell pairs found.")
            return

        # Calculate raw trade returns
        trade_returns = trades['Sell'] - trades['Buy']

        # Apply fee on both buy and sell transactions
        net_returns = ((trade_returns - (trades['Buy'] * self.buy_fee_percent) - (trades['Sell'] * self.sell_fee_percent)) / trades['Buy']) * 100

        # Store the returns for each trade
        self.returns = net_returns.tolist()

    def print_returns(self):
        """
        Print each trade's return and the total return.
        """
        if not self.returns:
            print("No complete buy-sell pairs found.")
            return
        
        for i, trade_return in enumerate(self.returns, 1):
            print(f"Trade {i}: Return = {trade_return:.2f}%")
        
        total_return = np.sum(self.returns)  # Sum of all returns
        print(f"Total Return for all trades with fees using traditional MACD: {total_return:.2f}%")

    def get_total_return(self):
        """
        Return the sum of all returns for the trades.
        """
        if not self.returns:
            return 0
        return round(np.sum(self.returns),2)




class KF:
    def __init__(self, data, price_column = 'close', process_variance = 0.2, measurement_vairance= 0.2 ):

        self.data = data
        self.price_column = price_column   # x
        self.process_variance = process_variance # Q
        self.measurement_variance = measurement_vairance # R

        self.data["KalmanFiltered"] = np.nan

    def apply_kalman_filter(self):
        price_data = self.data[self.price_column].values

        # Initialize Kalman filter parameters
        x_est_last = price_data[0]  # Initial estimate using first price value
        P_last = 1.0  # Initial estimate of error covariance (set to high value for uncertainty)

        # Preallocate memory for the filtered output to optimize performance
        kalman_filtered_data = np.zeros_like(price_data)

        for i in range(1, len(price_data)):
            # Prediction step
            x_temp_est = x_est_last
            P_temp = P_last + self.process_variance
            
            # Kalman Gain
            K = P_temp / (P_temp + self.measurement_variance)
            
            # Update estimate with measurement
            x_est = x_temp_est + K * (price_data[i] - x_temp_est)
            
            # Update error covariance
            P = (1 - K) * P_temp
            
            # Store the filtered value
            kalman_filtered_data[i] = x_est
            
            # Prepare for next iteration
            x_est_last = x_est
            P_last = P

        # Store the Kalman-filtered data in the DataFrame
        self.data['KalmanFiltered'] = kalman_filtered_data

    def get_filtered_data(self):
        """
        Return the DataFrame with the original and Kalman-filtered data.
        """
        return self.data[[self.price_column, 'KalmanFiltered']]
    
    
    def plot_results(self, starting_index = 1, end_index = None):
        """
        Plot the original price and Kalman-filtered data for visualization.
        """
        if end_index is None:
            end_index = len(self.data)
            
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index[starting_index:end_index], self.data[self.price_column][starting_index:end_index], label='Original Price', color='blue')
        plt.plot(self.data.index[starting_index:end_index], self.data['KalmanFiltered'][starting_index:end_index], label='Kalman Filtered', color='green')
        plt.legend()
        plt.title('Original Price vs. Kalman Filtered Price')
        plt.show()




class KalmanFilterEM:
    def __init__(self, data, price_column, max_iter=100, tol=1e-3):
        """
        Initialize the Kalman Filter with EM algorithm class.
        
        :param data: DataFrame containing stock price data.
        :param price_column: The column name containing the price data.
        :param max_iter: Maximum number of iterations for the EM algorithm (default=100).
        :param tol: Convergence tolerance for the EM algorithm (default=1e-3).
        """
        self.data = data
        self.price_column = price_column
        self.max_iter = max_iter
        self.tol = tol
        
        self.price_data = self.data[self.price_column].values
        self.N = len(self.price_data)
        
        # Initialize Q (process noise variance) and R (measurement noise variance)
        self.Q = 1.0  # Initial guess for process noise variance
        self.R = 1.0  # Initial guess for measurement noise variance

    def kalman_filter(self):
        """
        Perform the forward pass of the Kalman filter.
        Returns the filtered state estimates and error covariances.
        """
        x_est = np.zeros(self.N)  # Estimated states
        P_est = np.zeros(self.N)  # Error covariances
        
        # Initial guesses for the first point
        x_est[0] = self.price_data[0]  # Initial state
        P_est[0] = 1.0  # Initial covariance

        # Kalman filtering process
        for t in range(1, self.N):
            # Prediction
            x_pred = x_est[t-1]  # Predict the next state
            P_pred = P_est[t-1] + self.Q  # Predict the error covariance
            
            # Kalman gain
            K = P_pred / (P_pred + self.R)
            
            # Update
            x_est[t] = x_pred + K * (self.price_data[t] - x_pred)
            P_est[t] = (1 - K) * P_pred
        
        return x_est, P_est

    def kalman_smoother(self, x_est, P_est):
        """
        Perform the backward pass (smoother) to refine state estimates.
        Returns the smoothed state estimates.
        """
        x_smooth = np.copy(x_est)  # Initialize smoothed estimates
        P_smooth = np.copy(P_est)  # Initialize smoothed error covariances
        
        for t in range(self.N-2, -1, -1):
            # Smoothing gain
            C = P_est[t] / (P_est[t] + self.Q)
            
            # Update smoothed estimates
            x_smooth[t] = x_est[t] + C * (x_smooth[t+1] - x_est[t])
            P_smooth[t] = P_est[t] + C * (P_smooth[t+1] - P_est[t])
        
        return x_smooth, P_smooth

    def optimize_em(self):
        """
        Optimize Q and R using the EM algorithm.
        """
        prev_log_likelihood = -np.inf
        
        for iteration in range(self.max_iter):
            # E-step: Run Kalman filter and smoother
            x_est, P_est = self.kalman_filter()  # Forward pass (filter)
            x_smooth, P_smooth = self.kalman_smoother(x_est, P_est)  # Backward pass (smoother)
            
            # M-step: Update Q and R
            # Process noise variance (Q) update
            Q_new = np.sum((x_smooth[1:] - x_smooth[:-1])**2 + P_smooth[1:]) / (self.N - 1)
            
            # Measurement noise variance (R) update
            R_new = np.sum((self.price_data - x_smooth)**2 + P_smooth) / self.N
            
            # Check for convergence (based on log-likelihood)
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * (P_smooth + self.R)) + ((self.price_data - x_smooth)**2) / (P_smooth + self.R))
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f'Converged after {iteration+1} iterations.')
                break
            
            # Update Q, R, and log-likelihood
            self.Q = Q_new
            self.R = R_new
            prev_log_likelihood = log_likelihood
            
        print(f"Final Q: {self.Q}, Final R: {self.R}")
        return self.Q, self.R

    def get_smoothed_data(self):
        """
        Get the smoothed data using the optimized Q and R.
        """
        x_est, P_est = self.kalman_filter()
        x_smooth, P_smooth = self.kalman_smoother(x_est, P_est)
        return x_smooth

    def plot_results(self, start, end):
        """
        Plot the original price and Kalman-filtered smoothed data.
        """
        smoothed_data = self.get_smoothed_data()  # Get smoothed data with optimized Q and R
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.data.index[start: end], self.price_data[start: end], label='Original Price', color='blue')
        plt.plot(self.data.index[start: end], smoothed_data[start: end], label='Smoothed Data (Kalman Filter)', color='green')
        plt.legend()
        plt.title('Original Price vs. Kalman Filter Smoothed Price')
        plt.show()



class MACDBacktester:
    def __init__(self, data, fast_ema, slow_ema, signal_line, signal_price = 'close', real_price = 'close',
                  sell_fee = 0.115, buy_fee = 0.115, initial_capital = 100):
        self.data = data.copy()
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.signal_line = signal_line
        self.signal_price = signal_price
        self.real_price = real_price
        self.sell_fee_percent = sell_fee / 100
        self.buy_fee_percent = buy_fee / 100
        self.initial_capital = initial_capital
        self.trades = []
        self.data['positions'] = 0
        
    def calculate_macd(self):
        self.data['fast_ema'] = self.data[self.signal_price].ewm(span = self.fast_ema, adjust = False).mean()
        self.data['slow_ema'] = self.data[self.signal_price].ewm(span = self.slow_ema, adjust = False).mean()
        self.data['macd_line'] = self.data['fast_ema'] - self.data['slow_ema']
        self.data['signal_line'] = self.data['macd_line'].ewm(span = self.signal_line, adjust = False).mean()


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
        annualized_return = ((1 + returns.mean()) ** 252 - 1) * 100  # Assuming daily returns
        annualized_volatility = returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else np.nan
        max_drawdown = ((self.results['total'].cummax() - self.results['total']) / self.results['total'].cummax()).max() * 100

        metrics = {
            'Total Return (%)': total_return,
            'Annualized Return (%)': annualized_return,
            'Annualized Volatility (%)': annualized_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown (%)': max_drawdown,
            'Win Rate (%)' : self.win_rate
        }
        return metrics

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