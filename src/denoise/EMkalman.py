import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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