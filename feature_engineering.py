import yfinance as yf
import numpy as np
import ta

def get_stock_data(ticker_symbol, period='20y',interval='1d'):
    # Fetch data for the last 20 years
    data = yf.download(ticker_symbol, period=period, interval='1d')
    columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    predict_data = data[columns].copy()
    predict_data.reset_index(drop=True, inplace=True)

    predict_data.columns = ['open', 'high', 'low', 'close', 'volume']
    df = predict_data

    # Calculate technical indicators
    df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    df['k_bar'] = (df['close'] - df['open']) / (df['high'] - df['low'])
    df['macd_DIF'] = ta.trend.macd(df['close'])
    df['macd'] = ta.trend.macd_diff(df['close']) * 2
    df['EMA'] = ta.trend.ema_indicator(df['close'], window=20) - ta.trend.ema_indicator(df['close'], window=10)
    df['EMA50'] = ta.trend.ema_indicator(df['close'], window=50)
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)

    # Calculate KD and J
    KKK, DDD, JJJ = calculate_kd(df['close'], df['high'], df['low'])
    df['KD'] = np.array(KKK) - np.array(DDD)
    df['J'] = JJJ

    # Calculate DMA
    df['DMA'] = calculate_DMA(df)

    # Reorder columns
    new_order = ['EMA', 'DMA', 'KD', 'J', 'EMA50', 'rsi_14', 'OBV', 'macd_DIF', 'macd', 'k_bar', 'open', 'high', 'low', 'close']
    df = df[new_order]

    # Return training data starting from index 50
    training_data = df.loc[50:]
    return training_data

def calculate_kd(close, high, low, window=9, smooth_window=3):
    k_values, d_values, j_values = [], [], []
    for i in range(len(close)):
        if i < window:
            k_values.append(0)
            d_values.append(0)
            j_values.append(0)
            continue

        highest_high = high[i - window + 1:i + 1].max()
        lowest_low = low[i - window + 1:i + 1].min()
        rsv = (close[i] - lowest_low) / (highest_high - lowest_low) * 100

        k = (smooth_window - 1) / smooth_window * k_values[-1] + 1 / smooth_window * rsv
        d = (smooth_window - 1) / smooth_window * d_values[-1] + 1 / smooth_window * k
        j = 3 * k - 2 * d

        k_values.append(k)
        d_values.append(d)
        j_values.append(j)
    return k_values, d_values, j_values

def calculate_DMA(df, window=5, smooth_window=20):
    MA_short = ta.trend.sma_indicator(df['close'], window)
    MA_long = ta.trend.sma_indicator(df['close'], smooth_window)

    DMA = MA_short - MA_long
    AMA = ta.trend.sma_indicator(DMA, 5)
    return DMA - AMA


def create_dataset(data, window_size=10, future_window=5):
    max_index = len(data) - window_size - future_window
    train_data, price = [], []
    for i in range(max_index + 1):
        

        # Get  features
        window = data.iloc[i:(i + window_size)].copy()
        min_val = window.min()  # Min over all features
        max_val = window.max()  # Max over all features
        window_scaled = ( ((window - min_val))/ (max_val - min_val) ) 
        #window_scaled = window_scaled.transpose()
        train_data.append(window_scaled) 
                        
        future_close_prices = data['close'].iloc[i + window_size  : i + window_size + future_window]
        future_close_prices_scaled = ( (future_close_prices - min_val['close']) / (max_val['close'] - min_val['close']) ) 
        price.append(future_close_prices_scaled) 
        
    return np.array(train_data), np.array(price)


