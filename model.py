from keras.engine.base_layer import Layer
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, LSTM, Dense, Dropout, Embedding
import config

def create_rnn_model(lstm_input_shape, average_label, lstm_cells = config.LSTM_CELLS, lstm_layers = config.LSTM_LAYERS,
 embedding_layer = False):

    input = Input(shape=lstm_input_shape)

    lstm_layer = LSTM(config.LSTM_CELLS)(input)

    output = Dense(1, activation=config.OUTPUT_ACTIVATION, bias_initializer=keras.initializers.Constant(average_label))(lstm_layer)

    model = keras.Model(inputs = input, outputs = output)

    # model = keras.Sequential()

    # if embedding_layer:
    #     model.add(
    #         Embedding(
    #             input_dim=19,
    #             output_dim=8,
    #             trainable=True
    #     ))

    # if lstm_layers > 1:
    #     for i in range(lstm_layers - 1):
    #         model.add(
    #             LSTM(
    #                 lstm_cells,
    #                 input_shape = lstm_input_shape,
    #                 return_sequences=True,
    #                 dropout=0.0,
    #                 recurrent_dropout=0.0
    #             )
    #         )
    #     lstm_input_shape = (2, config.LSTM_CELLS)
    # # Add final (or only) LSTM layer
    # model.add(LSTM(
    #         config.LSTM_CELLS,
    #         input_shape = lstm_input_shape,
    #         return_sequences=False,
    #         dropout=0.0,
    #         recurrent_dropout=0.0
    # ))

    # # Add final dense layer
    # model.add(Dense(
    #     units=1,
    #     activation=config.OUTPUT_ACTIVATION,
    #     bias_initializer=keras.initializers.Constant(average_label)
    # ))

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    return model

model = create_rnn_model((2,14), 0.5)
model.summary()