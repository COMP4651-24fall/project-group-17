from flask import Flask, render_template, jsonify, request
from feature_engineering import get_stock_data
from constant import sp500_ticker_symbol
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

def predict_stock(ticker_symbol):
    try:
        training_data = get_stock_data(ticker_symbol=ticker_symbol, period='6mo')
        training_data = training_data.tail(26)
        model_loaded = load_model(f'./model/{ticker_symbol}.h5')

        new_order = ['EMA', 'DMA', 'KD', 'J', 'EMA50', 'rsi_14', 'OBV', 'macd_DIF', 'macd', 'k_bar', 'open', 'high', 'low', 'close']
        df = training_data[new_order]
        min_val = df.min()
        max_val = df.max()
        window_scaled = (((df - min_val))/ (max_val - min_val))
        model_predict_data = np.array(window_scaled)

        predicted_prices = model_loaded.predict(model_predict_data[np.newaxis,:])
        predicted_prices_normal = np.array(((predicted_prices) * (max_val["close"] - min_val["close"]) + min_val["close"])[0])
        current_price = df["close"].tail(1).iloc[0]
        percentage_change = ((predicted_prices_normal[4] - current_price)/current_price) * 100.0

        return {
            "success": True,
            "current_price": round(float(current_price), 2),
            "predicted_price": round(float(predicted_prices_normal[4]), 2),
            "percentage_change": round(float(percentage_change), 2)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/')
def home():
    return render_template('index.html', symbols=sp500_ticker_symbol)

@app.route('/api/predict/<ticker_symbol>')
def predict(ticker_symbol):
    if ticker_symbol not in sp500_ticker_symbol:
        return jsonify({"success": False, "error": "Invalid ticker symbol"})
    
    result = predict_stock(ticker_symbol)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 