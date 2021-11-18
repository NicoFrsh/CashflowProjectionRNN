import numpy as np
import tensorflow as tf
import keras
from keras.layers import LSTM, Dense, Dropout, Embedding
import config

def create_rnn_model(lstm_input_shape, lstm_cells = config.LSTM_CELLS, lstm_layers = config.LSTM_LAYERS, embedding_layer = False):

    model = keras.Sequential()

    if embedding_layer:
        model.add(
            Embedding(
                input_dim=19,
                output_dim=8,
                trainable=True
        ))

    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            model.add(
                LSTM(
                    lstm_cells,
                    return_sequences=True,
                    dropout=0.1,
                    recurrent_dropout=0.1
                )
            )
    # Add final (or only) LSTM layer
    model.add(LSTM(
            lstm_cells,
            input_shape = lstm_input_shape,
            return_sequences=False,
            dropout=0.1,
            recurrent_dropout=0.1
    ))

    # Add final dense layer
    model.add(Dense(
        units=1,
        activation=config.OUTPUT_ACTIVATION
    ))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model