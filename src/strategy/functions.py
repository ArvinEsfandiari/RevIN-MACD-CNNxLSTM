import pandas as pd
import numpy as np

def macd_crossover_strategy(df, macd_col='macd_close_28', signal_col='signal_close_28' ):
    buy_signals = (df[macd_col] > df[signal_col]) & (df[macd_col].shift(1) <= df[signal_col].shift(1))
    sell_signals = (df[macd_col] < df[signal_col]) & (df[macd_col].shift(1) >= df[signal_col].shift(1))

    signals = np.zeros((len(df),1), dtype=np.int8)
    signals[buy_signals] = 1
    signals[sell_signals] = -1

    # Condition to buy first. It is considered that we just have long positions.
    trade_idx = np.where(signals !=0)[0]

    if len(trade_idx)>0 and signals[trade_idx[0]]==-1: # If we have a trade and it is sell then:
        signals[trade_idx[0]] = 0
    
    # If you have any open position, close it.
    if len(trade_idx)>0 and signals[trade_idx[-1]] ==1:
        signals[-1] = -1

    df.loc[:,'signals'] = signals
    return df


