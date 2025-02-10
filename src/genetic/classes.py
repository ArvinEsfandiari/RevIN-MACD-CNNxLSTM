import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pygad
import warnings


# Suppress specific warnings (UserWarnings in this case)

# New backtester
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
        """ Backtests the trading strategy. """
        self.data = self.data.copy()
        self.data['price'] = self.data[self.real_price]
        self.data['positions_diff'] = self.data['positions'].diff().fillna(0)
        
        self.data['cash'] = self.initial_capital
        self.data['holdings'] = 0.0
        self.data['total'] = self.initial_capital

        cash = self.initial_capital
        holdings = 0.0
        position = False
        buy_price = 0.0
        win_count = 0
        total_trades = 0

        for idx, row in self.data.iterrows():
            position_change = row['positions_diff']
            price = row['price']

            if position_change == 1 and not position:  # Enter long position
                holdings = cash * (1 - self.buy_fee_percent) / price
                cash = 0
                position = True
                buy_price = price

            elif position_change == -1 and position: # Exit long position
                cash = holdings * price * (1 - self.sell_fee_percent)
                holdings = 0
                position = False
                trade_return = ((price - buy_price) / buy_price) * 100
                self.trades.append(trade_return)
                total_trades += 1
                if trade_return > 0:
                    win_count += 1

            total = cash + (holdings * price if position else 0)
            self.data.at[idx, 'cash'] = float(cash)
            self.data.at[idx, 'holdings'] = float(holdings * price if position else 0)
            self.data.at[idx, 'total'] = float(total)

        self.results = self.data[['cash', 'holdings', 'total']]
        self.win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        return self.data, self.trades


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


# Older backtester

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



import pygad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings

# Suppress specific warnings (UserWarnings in this case)
warnings.filterwarnings("ignore", category=UserWarning)


class MACD_GA:
    def __init__(self, df_train, fast_range=(1, 50), slow_range=(2, 50), signal_range=(2, 30),\
                  generations=120, population=50, mating_parents=25, real_price='close', signal_price = 'close' ):
        """"
        Find the best parameters for MACD with GA that has the best tuning.

        :param df_train: Dataframe containing dataset with or without smoothing column.
        :param fast_range: Tuple contains fast ema period.
        :param slow_range: Tuple contains slow ema period.
        :param signal_line: Tuple contains signal line period.
        :param generations: Integer contains the number generation in GA.
        :param population: Integer contains the number of solution in each generation.
        :param mating_parents: Integer contains the number of parents after choosing.
        :param real_price: Defualt is 'close' and is suitable for buy and sell.
        :param signal_price: Defualt is 'close' and is suitable for generation buy and sell signals.

        """
        self.df_train = df_train
        self.fast_range = fast_range
        self.slow_range = slow_range
        self.signal_range = signal_range
        self.generations = generations
        self.population = population
        self.mating_parents = mating_parents
        self.ga_instance = None
        self.best_solution = None
        self.real_price = real_price
        self.signal_price = signal_price


    def macd(self, data, fast, slow, signal):
        data['ema_fast'] = data[self.signal_price].ewm(span=fast, min_periods=1, adjust=False).mean()
        data['ema_slow'] = data[self.signal_price].ewm(span=slow, min_periods=1, adjust=False).mean()
        data['macd_line'] = data['ema_fast'] - data['ema_slow']
        data['signal_line'] = data['macd_line'].ewm(span=signal, adjust=False).mean()
        return data

    @staticmethod
    def backtest(data):
        buy_signals = (data['macd_line'] > data['signal_line']) & (data['macd_line'].shift(1) <= data['signal_line'].shift(1))
        sell_signals = (data['macd_line'] < data['signal_line']) & (data['macd_line'].shift(1) >= data['signal_line'].shift(1))

        positions = pd.Series(index=data.index, data=np.nan)
        positions[buy_signals] = 1
        positions[sell_signals] = 0
        positions.ffill(inplace=True)
        positions.fillna(0, inplace=True)

        data['Strategy'] = positions.shift(1) * (data['high'].pct_change())
        return data['Strategy'].cumsum().iloc[-1]

    def fitness_function(self, ga_instance, solution, solution_idx):
        ema_fast, ema_slow, signal = int(solution[0]), int(solution[1]), int(solution[2])
        if ema_fast >= ema_slow - 1:
            return -np.inf
        if ema_fast < 1 or ema_slow < ema_fast + 1 or signal < 1:
            return -np.inf
        temp_data = self.macd(self.df_train.copy(), ema_fast, ema_slow, signal)
        profit = self.backtest(temp_data)
        return profit

    def run_ga(self):
        # Initialize the GA instance
        self.ga_instance = pygad.GA(
            num_generations=self.generations,
            sol_per_pop=self.population,
            num_parents_mating=self.mating_parents,
            fitness_func=self.fitness_function,
            crossover_type="two_points",
            crossover_probability=0.8,
            num_genes=3,
            init_range_low=[self.fast_range[0], self.slow_range[0], self.signal_range[0]],
            init_range_high=[self.fast_range[1], self.slow_range[1], self.signal_range[1]],
            mutation_percent_genes=50,
            gene_type=int,
            save_solutions=True
        )
        # Run the GA
        self.ga_instance.run()
        # Save the best solution
        self.best_solution = self.ga_instance.best_solution()

    def get_best_parameters(self):
        if self.best_solution is None:
            return None
        solution, fitness, _ = self.best_solution
        return {
            "Fast EMA": int(solution[0]),
            "Slow EMA": int(solution[1]),
            "Signal Line": int(solution[2]),
            "Net Profit": fitness
        }

    def plot_results(self):
        if self.ga_instance:
            self.ga_instance.plot_fitness(title="Fitness Evolution")
            # self.ga_instance.plot_fitness()
            self.ga_instance.plot_genes(title="Fast, Slow, Signal")
            self.ga_instance.plot_new_solution_rate(title="New Solution Rate Evolution")



            

class MACD_GA_v2:
    def __init__(self, df_train, fast_range=(1, 5), slow_range=(2, 14), signal_range=(2, 7),\
                  generations=120, population=50, mating_parents=25, real_price='close', signal_price = 'close' ):
        """"
        Find the best parameters for MACD with GA that has the best tuning.

        :param df_train: Dataframe containing dataset with or without smoothing column.
        :param fast_range: Tuple contains fast ema period.
        :param slow_range: Tuple contains slow ema period.
        :param signal_line: Tuple contains signal line period.
        :param generations: Integer contains the number generation in GA.
        :param population: Integer contains the number of solution in each generation.
        :param mating_parents: Integer contains the number of parents after choosing.
        :param real_price: Defualt is 'close' and is suitable for buy and sell.
        :param signal_price: Defualt is 'close' and is suitable for generation buy and sell signals.

        """
        self.df_train = df_train
        self.fast_range = fast_range
        self.slow_range = slow_range
        self.signal_range = signal_range
        self.generations = generations
        self.population = population
        self.mating_parents = mating_parents
        self.ga_instance = None
        self.best_solution = None
        self.real_price = real_price
        self.signal_price = signal_price


    def macd(self, data, fast, slow, signal):
        data['ema_fast'] = data[self.signal_price].ewm(span=fast, min_periods=1, adjust=False).mean()
        data['ema_slow'] = data[self.signal_price].ewm(span=slow, min_periods=1, adjust=False).mean()
        data['macd_line'] = data['ema_fast'] - data['ema_slow']
        data['signal_line'] = data['macd_line'].ewm(span=signal, adjust=False).mean()
        return data

    @staticmethod
    def backtest(data):
        buy_signals = (data['macd_line'] > data['signal_line']) & (data['macd_line'].shift(1) <= data['signal_line'].shift(1))
        sell_signals = (data['macd_line'] < data['signal_line']) & (data['macd_line'].shift(1) >= data['signal_line'].shift(1))

        positions = pd.Series(index=data.index, data=np.nan)
        positions[buy_signals] = 1
        positions[sell_signals] = 0
        positions.ffill(inplace=True)
        positions.fillna(0, inplace=True)

        data['Strategy'] = positions.shift(1) * (data['close'].pct_change())
        return data['Strategy'].cumsum().iloc[-1]

    def fitness_function(self, ga_instance, solution, solution_idx):
        ema_fast, ema_slow, signal = int(solution[0]), int(solution[1]), int(solution[2])
        if ema_fast >= ema_slow - 1:
            return -np.inf
        if ema_fast < 1 or ema_slow < ema_fast + 1 or signal < 1:
            return -np.inf
        temp_data = self.macd(self.df_train.copy(), ema_fast, ema_slow, signal)
        profit = self.backtest(temp_data)
        return profit

    def run_ga(self):
        # Initialize the GA instance
        self.ga_instance = pygad.GA(
            num_generations=self.generations,
            sol_per_pop=self.population,
            num_parents_mating=self.mating_parents,
            fitness_func=self.fitness_function,
            crossover_type="two_points",
            crossover_probability=0.8,
            num_genes=3,
            init_range_low=[self.fast_range[0], self.slow_range[0], self.signal_range[0]],
            init_range_high=[self.fast_range[1], self.slow_range[1], self.signal_range[1]],
            mutation_percent_genes=50,
            gene_type=int,
            save_solutions=True
        )
        # Run the GA
        self.ga_instance.run()
        # Save the best solution
        self.best_solution = self.ga_instance.best_solution()

    def get_best_parameters(self):
        if self.best_solution is None:
            return None
        solution, fitness, _ = self.best_solution
        return {
            "Fast EMA": int(solution[0]),
            "Slow EMA": int(solution[1]),
            "Signal Line": int(solution[2]),
            "Net Profit": fitness
        }

    def plot_results(self):
        if self.ga_instance:
            self.ga_instance.plot_fitness(title="Fitness Evolution")
            # self.ga_instance.plot_fitness()
            self.ga_instance.plot_genes(title="Fast, Slow, Signal")
            self.ga_instance.plot_new_solution_rate(title="New Solution Rate Evolution")