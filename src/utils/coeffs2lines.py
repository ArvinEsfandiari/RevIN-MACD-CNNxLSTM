import pandas as pd
import numpy as np

def calculate_ema(values, period):
    return pd.Series(values).ewm(span=period, adjust=False).mean().values

def calculate_macd(sequence, fast_period, slow_period, signal_period):
    # calculate macd things for a sequence.
    macd_line = calculate_ema(sequence, fast_period)-calculate_ema(sequence, slow_period)
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line-signal_line
    return macd_line, signal_line, histogram

def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length + 1):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences, dtype=np.int64)


def coeffs2lines(coeffs, sequences):
    macd_lines = []
    signal_lines = []
    histograms = []


    for i, seq in enumerate(sequences):
        f,s,h = coeffs[i]
        macd_line, signal_line, histogram = calculate_macd(seq, fast_period=f,slow_period=s,signal_period=h)
        macd_lines.append(macd_line)
        signal_lines.append(signal_line)
        histograms.append(histogram)

    macd_lines = np.array(macd_lines)
    signal_lines = np.array(signal_lines)
    histograms = np.array(histograms)

    return macd_lines, signal_lines, histograms

def label_sequence(macd_lines, signal_lines ):

    a, b = macd_lines.shape
    labels = np.zeros((a,b), dtype=np.int8)
    for i in range(0,a):
        # a = 3931
        for j in range(2, b):
            # b = 120

            if macd_lines[i,j] > signal_lines[i,j] and macd_lines[i, j-1]<=signal_lines[i, j-1]:
                labels[i, j] = 1 # Buy
            elif macd_lines[i,j] < signal_lines[i,j] and macd_lines[i, j-1] >=signal_lines[i,j-1]:
                labels[i,j] = -1 # Sell

    return labels