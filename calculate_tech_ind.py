import numpy as np
import pandas as pd

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    """Calculates MACD"""
    short_ema = np.array(data['Close'].ewm(span=short_window, adjust=False).mean())
    long_ema = np.array(data['Close'].ewm(span=long_window, adjust=False).mean())
    data['MACD'] = short_ema - long_ema
    return data

def calculate_rsi(data, window=14):
    """Calculates Relative Strength Index (RSI)"""
    delta = data['Close'].diff()

    # Calculate gains and losses
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()

    #print(gain)
    #print(loss)
    # Handle cases where loss is zero
    rs = gain / loss
   
    rs[loss == 0] = np.inf  # If loss is zero, set RS to infinity
    rs[gain == 0] = 0       # If gain is zero and loss is not zero, set RS to 0
    rs = np.array(rs)
    data['RSI'] = 100 - (100 / (1 + rs))
    # For RS=inf, RSI should be 100
    return data

def calculate_cci(data, window=20):
    """Calculates Commodity Channel Index (CCI)."""
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    n_tp = np.array(tp)
    #print(tp)
    ma = np.array(tp.rolling(window=window).mean())
    #print(ma)
    mad = np.array(tp.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean()))
    #print(mad)
    data['CCI'] = (n_tp - ma) / (0.015 * mad)
    #print(data['CCI'])
    return data
