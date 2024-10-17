import pandas as pd
import numpy as np
import pygad

class MACDOptimizer:
    def __init__(self, data_file, price_type='close'):
        """
        Initializes the MACDOptimizer class.

        Parameters:
        - data_file: Path to the CSV file containing the data.
        - price_type: The price column to use ('open', 'high', 'low', 'close').
        """
        self.data_file = data_file
        self.price_type = price_type
        self.data = self.load_data()

    def load_data(self):
        """
        Loads data from the input, which could be a CSV file path or a pandas DataFrame.
        """
        if isinstance(self.data_file, pd.DataFrame):
            full_data = self.data_file
        else:
            full_data = pd.read_csv(self.data_file)
            full_data.set_index('time', inplace=True)
        return full_data

    def calculate_macd(self, data, fast_period, slow_period, signal_period):
        """
        Calculates the MACD indicator.

        Parameters:
        - data: The DataFrame containing the price data.
        - fast_period: The period for the fast EMA.
        - slow_period: The period for the slow EMA.
        - signal_period: The period for the signal line EMA.
        """
        # Compute the EMAs
        ema_fast = data[self.price_type].ewm(span=fast_period, adjust=False).mean()
        ema_slow = data[self.price_type].ewm(span=slow_period, adjust=False).mean()
        # Calculate MACD components
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        # Add to DataFrame
        data['macd_line'] = macd_line
        data['signal_line'] = signal_line
        data['histogram'] = histogram
        return data

    def generate_signals(self, data):
        """
        Generates trading signals based on MACD crossover.

        Parameters:
        - data: The DataFrame containing the MACD data.
        """
        data['signal'] = 0
        # Corrected assignment to avoid SettingWithCopyWarning
        data.loc[data.index[1:], 'signal'] = np.where(
            data['macd_line'].iloc[1:] > data['signal_line'].iloc[1:], 1, 0
        )
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

    def optimize(self):
        """Runs the genetic algorithm to optimize MACD parameters."""

        def fitness_func(ga_instance, solution, solution_idx):
            """
            Fitness function for the genetic algorithm.

            Parameters:
            - ga_instance: The instance of the GA.
            - solution: The array of MACD parameters.
            - solution_idx: The index of the solution.
            """
            try:
                fast_period = int(solution[0])
                slow_period = int(solution[1])
                signal_period = int(solution[2])

                # Ensure valid periods
                if fast_period >= slow_period or fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
                    return -np.inf  # Invalid solution

                # Copy data to avoid modifying the original
                data = self.data.copy()
                data = self.calculate_macd(data, fast_period, slow_period, signal_period)
                data = self.generate_signals(data)
                total_return = self.backtest_strategy(data)
                return total_return

            except Exception as e:
                print(f"Error in fitness function at solution index {solution_idx}: {e}")
                return -np.inf

        gene_space = [
            {'low': 1, 'high': 50, 'step': 1},    # fast_period
            {'low': 5, 'high': 50, 'step': 1},   # slow_period
            {'low': 1, 'high': 30, 'step': 1},    # signal_period
        ]

        ga_instance = pygad.GA(
            num_generations=50,
            num_parents_mating=5,
            fitness_func=fitness_func,
            sol_per_pop=20,
            num_genes=3,
            gene_space=gene_space,
            parent_selection_type="sss",
            keep_parents=2,
            mutation_type="random",
            mutation_num_genes=1,  # Mutate 1 gene per solution
            #mutation_probability=0.1,
            #crossover_probability = 0.9
            #delay_after_gen=0   # Avoid deprecation warning
        )

        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        print("Best MACD Parameters:")
        print(f"Fast Period: {int(solution[0])}")
        print(f"Slow Period: {int(solution[1])}")
        print(f"Signal Period: {int(solution[2])}")
        print("Total Profit from Best Solution:", solution_fitness)
        return solution, solution_fitness

# Usage Example:
# import time
# start_time = time.time()
# optimizer = MACDOptimizer(data_file="data/BTCUSD_1m_2024-09-23.csv", price_type='close')
# best_solution, best_profit = optimizer.optimize()
# end_time = time.time()
# print(f"Running time is:{end_time-start_time} seconds.")

