import pandas as pd
import yfinance as yf
import numpy as np
import math

# parameter setup (default values in the original indicator)
length = 20
mult = 2
length_KC = 20
mult_KC = 1.5

# get asset prices
df = yf.download(tickers='BTC-USD', period='22h', interval='1m')


# moving average
def sma(dataframe, length: int = 50):
    dataframe[f"SMA_{length}"] = df['close'].rolling(window=length).mean()
    return dataframe


def bbands(dataframe, length: int, multiplier: float):
    m_avg = dataframe['close'].rolling(window=length).mean()
    m_std = dataframe['close'].rolling(window=length).std(ddof=0)
    dataframe['upper_BB'] = m_avg + multiplier * m_std
    dataframe['lower_BB'] = m_avg - multiplier * m_std
    return dataframe


def keltner(dataframe, length: int, mulitplier: float):
    m_avg = dataframe['close'].rolling(window=length).mean()
    rng_ma = dataframe['tr'].rolling(window=length).mean()
    dataframe['upper_KC'] = m_avg + rng_ma * mulitplier
    dataframe['lower_KC'] = m_avg - rng_ma * mulitplier
    return dataframe


def squeeze(dataframe, length, multiplier, length_kc: int, mult_kc: float):
    # moving average
    m_avg = dataframe['close'].rolling(window=length).mean()
    # standard deviation
    m_std = dataframe['close'].rolling(window=length).std(ddof=0)
    # true_range
    dataframe['tr0'] = abs(dataframe['high'] - dataframe['low'])
    dataframe['tr1'] = abs(dataframe['high'] - dataframe['close'].shift())
    dataframe['tr2'] = abs(dataframe['low'] - dataframe['close'].shift())
    dataframe['tr'] = dataframe[['tr0', 'tr1', 'tr2']].max(axis=1)
    # moving average of TR
    range_ma = dataframe['tr'].rolling(window=length_KC).mean()

    ## Bollinger Bands
    dataframe['upper_BB'] = m_avg + mult * m_std
    dataframe['lower_BB'] = m_avg - mult * m_std

    # Keltner Channel
    dataframe['upper_KC'] = m_avg + range_ma * mult_KC
    dataframe['lower_KC'] = m_avg - range_ma * mult_KC

    # SQUEEZE INDICATOR
    # Bar Value
    highest = dataframe['high'].rolling(window=length_KC).max()
    lowest = dataframe['low'].rolling(window=length_KC).min()
    m1 = (highest + lowest) / 2
    dataframe['value'] = (dataframe['close'] - (m1 + m_avg) / 2)
    fit_y = np.array(range(0, length_KC))
    dataframe['value'] = dataframe['value'].rolling(window=length_KC).apply(
        lambda x: np.polyfit(fit_y, x, 1)[0] * (length_KC - 1) + np.polyfit(fit_y, x, 1)[1], raw=True)
    # check for squeeze
    dataframe['sqz_on'] = (dataframe['lower_BB'] > dataframe['lower_KC']) & (
            dataframe['upper_BB'] < dataframe['upper_KC'])
    dataframe['sqz_off'] = (dataframe['lower_BB'] < dataframe['lower_KC']) & (
            dataframe['upper_BB'] > dataframe['upper_KC'])
    # buying window for long position:
    # 1. black cross becomes gray (the squeeze is released)
    long_cond1 = (dataframe['sqz_off'][-2] == False) & (dataframe['sqz_off'][-1] == True)
    # 2. bar value is positive => the bar is light green k
    long_cond2 = dataframe['value'][-1] > 0
    dataframe["sqz_long"] = long_cond1 and long_cond2
    # 1. black cross becomes gray (the squeeze is released)
    short_cond1 = (dataframe['squeeze_off'][-2] == False) & (dataframe['squeeze_off'][-1] == True)
    # 2. bar valueis negative => the bar is light red
    short_cond2 = df['value'][-1] < 0
    dataframe["sqz_short"] = short_cond1 and short_cond2
    return dataframe


