from keras.engine.base_layer import Layer
import numpy as np
import tensorflow as tf
import keras
from keras.layers import Input, LSTM, Dense, Dropout, Embedding
import config

def create_rnn_model(lstm_input_shape, average_label, average_label_2 = None, lstm_cells = config.LSTM_CELLS, lstm_layers = config.LSTM_LAYERS,
 embedding_layer = False):

   input = Input(shape=lstm_input_shape)

   lstm_layer = LSTM(config.LSTM_CELLS)(input)

   net_profit_head = Dense(1, activation=config.OUTPUT_ACTIVATION, bias_initializer=keras.initializers.Constant(average_label),
   name = 'net_profit_head')(lstm_layer)

   if config.USE_ADDITIONAL_INPUT: # Additional network head for additional input
      additional_output_head = Dense(1, activation=config.OUTPUT_ACTIVATION, bias_initializer=keras.initializers.Constant(average_label_2),
      name = 'additional_input_head')(lstm_layer)
   
      model = keras.Model(inputs = input, outputs =[net_profit_head, additional_output_head])

   else:
      model = keras.Model(inputs = input, outputs = net_profit_head)

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

   if config.USE_ADDITIONAL_INPUT:
      model.compile(optimizer='adam', loss={'net_profit_head':'mse','additional_input_head':'mse'}, 
      loss_weights={'net_profit_head': 0.5, 'additional_input_head': 0.5})
   else:
      model.compile(optimizer='adam', loss='mse', metrics=['mae'])

   return model