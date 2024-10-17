import pandas as pd
import numpy as np
import pygad
import torch
import torch.nn as nn
import torch.optim as optim

class GAParameterOptimizer(nn.Module):
    def __init__(self, input_size, output_size):
        super(GAParameterOptimizer, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, output_size)
        self.sigmoid = nn.Sigmoid()  # To keep outputs in [0,1]

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        return out

class MACDOptimizerGA_new:
    def __init__(self, data_file, price_type='close'):
        """
        Initializes the MACDOptimizer class.

        Parameters:
        - data_file: Path to the CSV file containing the data or a pandas DataFrame.
        - price_type: The price column to use ('open', 'high', 'low', 'close').
        """
        self.data_file = data_file
        self.price_type = price_type
        self.data = self.load_data()

        # Initialize neural network for GA parameter optimization
        self.ga_param_nn = GAParameterOptimizer(input_size=1, output_size=5)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.ga_param_nn.parameters(), lr=0.001)
        self.fitness_history = []
        self.ga_params_history = []

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
        data['signal'][1:] = np.where(
            data['macd_line'][1:] > data['signal_line'][1:], 1, 0
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
        """Runs the genetic algorithm to optimize MACD parameters and GA parameters."""

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

        # Initial GA parameters
        for iteration in range(1):
            # Prepare input for neural network (e.g., previous best fitness)
            if self.fitness_history:
                nn_input = torch.tensor([[self.fitness_history[-1]]], dtype=torch.float32)
            else:
                nn_input = torch.tensor([[0.0]], dtype=torch.float32)

            # Neural network predicts GA parameters
            ga_params_normalized = self.ga_param_nn(nn_input).detach().numpy()[0]
            # Denormalize GA parameters to actual values
            num_generations = int(ga_params_normalized[0] * 100) + 10  # Range: 10 to 110
            sol_per_pop = int(ga_params_normalized[1] * 50) + 10        # Range: 10 to 60
            num_parents_mating = int(ga_params_normalized[2] * (sol_per_pop - 2)) + 2  # At least 2 parents
            mutation_probability = ga_params_normalized[3] * 0.5 + 0.01  # Range: 0.01 to 0.51
            crossover_probability = ga_params_normalized[4] * 0.5 + 0.5  # Range: 0.5 to 1.0

            # Ensure valid GA parameters
            num_generations = max(10, num_generations)
            sol_per_pop = max(10, sol_per_pop)
            num_parents_mating = min(num_parents_mating, sol_per_pop)
            mutation_probability = np.clip(mutation_probability, 0.01, 0.5)
            crossover_probability = np.clip(crossover_probability, 0.5, 1.0)

            # Record GA parameters
            ga_params = {
                'num_generations': num_generations,
                'sol_per_pop': sol_per_pop,
                'num_parents_mating': num_parents_mating,
                'mutation_probability': mutation_probability,
                'crossover_probability': crossover_probability
            }
            self.ga_params_history.append(ga_params)

            print(f"Iteration {iteration + 1}: GA Parameters:")
            print(ga_params)

            gene_space = [
                {'low': 1, 'high': 50, 'step': 1},   # fast_p eriod
                {'low': 5, 'high': 50, 'step': 1},   # slow_period
                {'low': 1, 'high': 30, 'step': 1},   # signal_period
            ]

            ga_instance = pygad.GA(
                num_generations=ga_params['num_generations'],
                num_parents_mating=ga_params['num_parents_mating'],
                fitness_func=fitness_func,
                sol_per_pop=ga_params['sol_per_pop'],
                num_genes=3,
                gene_space=gene_space,
                parent_selection_type="sss",
                keep_parents=2,
                crossover_type="single_point",
                mutation_type="random",
                mutation_probability=ga_params['mutation_probability'],
                crossover_probability=ga_params['crossover_probability'],
                mutation_num_genes=1,
            )

            ga_instance.run()

            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            print("Best MACD Parameters:")
            print(f"Fast Period: {int(solution[0])}")
            print(f"Slow Period: {int(solution[1])}")
            print(f"Signal Period: {int(solution[2])}")
            print("Total Profit from Best Solution:", solution_fitness)

            # Record fitness
            self.fitness_history.append(solution_fitness)

            # Prepare training data for neural network
            if len(self.fitness_history) > 1:
                # Input: Previous fitness
                inputs = torch.tensor([[self.fitness_history[-2]]], dtype=torch.float32)
                # Target: Current GA parameters (normalized)
                targets = torch.tensor([[
                    (ga_params['num_generations'] - 10) / 100,
                    (ga_params['sol_per_pop'] - 10) / 50,
                    (ga_params['num_parents_mating'] - 2) / (ga_params['sol_per_pop'] - 2),
                    (ga_params['mutation_probability'] - 0.01) / 0.5,
                    (ga_params['crossover_probability'] - 0.5) / 0.5
                ]], dtype=torch.float32)

                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward pass
                outputs = self.ga_param_nn(inputs)
                # Compute loss
                loss = self.criterion(outputs, targets)
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

        # Return the best solution from the last GA run
        return solution, solution_fitness
