from feature_engineering import get_stock_data,create_dataset
from constant import sp500_ticker_symbol
import numpy as np
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Conv1D,Reshape,MaxPooling1D,Flatten,Bidirectional,AveragePooling1D 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.util import nest
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense, Dropout, Layer
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras.layers import Layer
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import Attention, MultiHeadAttention
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


def build_model(input_shape):
    inputs = Input(shape=input_shape)
    conv1 = Conv1D(filters=36, kernel_size=6, activation='elu', padding='valid')(inputs)
    conv2 = Conv1D(filters=108, kernel_size=6, activation='elu', padding='valid')(conv1)
    conv3 = Conv1D(filters=256, kernel_size=6, activation='elu', padding='valid')(conv2)
    maxpool = MaxPooling1D(pool_size=4, strides=1, padding='valid')(conv3)
     
    lstm_attention_1 = Bidirectional(LSTM(108, return_sequences=True))(maxpool)

    attention_layer = Attention()([lstm_attention_1, lstm_attention_1])

    lstm_attention = Bidirectional(LSTM(36, return_sequences=False,kernel_regularizer=l1(l1=0.001) ))(attention_layer)
    outputs = Dense(5, activation='linear',kernel_regularizer=l2(l2=0.001) )(lstm_attention)  # Adjust based on your application (e.g., regression or classification)
   
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')  # Use the correct loss function for your problem
    return model


for ticker_symbol in sp500_ticker_symbol:

    model_path = f'./model/{ticker_symbol}.h5'
    
    if os.path.exists(model_path):
        print(f"Model for {ticker_symbol} already exists. Skipping...")
        continue
    
    try:
        training_data = get_stock_data(ticker_symbol)
        traning_data, price_data = create_dataset(training_data, window_size = 26, future_window=5)
        print(traning_data.shape)
        print(price_data.shape)
        train_data, val_data, train_labels, val_labels = train_test_split(traning_data, price_data, test_size=0.1 )
        model = build_model((train_data.shape[1], train_data.shape[2]))
        print(model.summary())

        history = model.fit(
            train_data, 
            train_labels, 
            epochs=60, 
            batch_size=32, 
            validation_data=(val_data, val_labels), 

        )

        model.save(f'./model/{ticker_symbol}.h5', save_format='h5')



    except Exception as e:
        continue

