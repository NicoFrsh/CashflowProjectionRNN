from keras.engine.base_layer import Layer
import keras
from keras.layers import Input, LSTM, SimpleRNN, Dense, Dropout, Embedding
import config

def create_rnn_model(rnn_input_shape, average_label, average_label_2 = None, embedding_layer = False):

   input = Input(shape=rnn_input_shape)

   current_output = input

   # TODO: NOT WORKING!!
   if config.LSTM_LAYERS > 1:
      for i in range(config.LSTM_LAYERS-1):
         current_output = SimpleRNN(config.LSTM_CELLS, return_sequences=True)(current_output)

   # encoder_lstm, state_h, state_c = LSTM(lstm_cells, return_sequences=True, return_state=True)(input)

   # encoder_states = [state_h, state_c]

   # decoder_lstm = LSTM(lstm_cells)(encoder_lstm, initial_state = encoder_states)

   current_output = SimpleRNN(config.LSTM_CELLS)(current_output)

   net_profit_head = Dense(1, activation=config.OUTPUT_ACTIVATION, bias_initializer=keras.initializers.Constant(average_label),
   name = 'net_profit_head')(current_output)

   if config.USE_ADDITIONAL_INPUT: # Additional network head for additional input
      additional_output_head = Dense(1, activation=config.ADDITIONAL_OUTPUT_ACTIVATION, bias_initializer=keras.initializers.Constant(average_label_2),
      name = 'additional_input_head')(current_output)
   
      model = keras.Model(inputs = input, outputs =[net_profit_head, additional_output_head])

   else:
      model = keras.Model(inputs = input, outputs = net_profit_head)

   if config.USE_ADDITIONAL_INPUT:
      model.compile(optimizer='adam', loss={'net_profit_head':'mse','additional_input_head':'mse'}, 
      loss_weights={'net_profit_head': 0.5, 'additional_input_head': 0.5})
   else:
      model.compile(optimizer='adam', loss='mse', metrics=['mae'])

   return model

def create_lstm_model(lstm_input_shape, average_label, average_label_2 = None, lstm_cells = config.LSTM_CELLS, lstm_layers = config.LSTM_LAYERS,
 embedding_layer = False):

   input = Input(shape=lstm_input_shape)

   current_output = input

   # TODO: NOT WORKING!!
   if lstm_layers > 1:
      for i in range(lstm_layers-1):
         current_output = LSTM(lstm_cells, return_sequences=True)(current_output)

   # encoder_lstm, state_h, state_c = LSTM(lstm_cells, return_sequences=True, return_state=True)(input)

   # encoder_states = [state_h, state_c]

   # decoder_lstm = LSTM(lstm_cells)(encoder_lstm, initial_state = encoder_states)

   current_output = LSTM(lstm_cells)(current_output)

   net_profit_head = Dense(1, activation=config.OUTPUT_ACTIVATION, bias_initializer=keras.initializers.Constant(average_label),
   name = 'net_profit_head')(current_output)

   if config.USE_ADDITIONAL_INPUT: # Additional network head for additional input
      additional_output_head = Dense(1, activation=config.ADDITIONAL_OUTPUT_ACTIVATION, bias_initializer=keras.initializers.Constant(average_label_2),
      name = 'additional_input_head')(current_output)
   
      model = keras.Model(inputs = input, outputs =[net_profit_head, additional_output_head])

   else:
      model = keras.Model(inputs = input, outputs = net_profit_head)

   if config.USE_ADDITIONAL_INPUT:
      model.compile(optimizer='adam', loss={'net_profit_head':'mse','additional_input_head':'mse'}, 
      loss_weights={'net_profit_head': 0.5, 'additional_input_head': 0.5})
   else:
      # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005), loss='mse', metrics=['mae'])
      model.compile(optimizer='adam', loss='mse', metrics=['mae'])
   # Two-headed architecture not programmable with sequential model!
   # model = keras.Sequential()

   # # if embedding_layer:
   # #    model.add(
   # #       Embedding(
   # #             input_dim=19,
   # #             output_dim=8,
   # #             trainable=True
   # #    ))

   # if lstm_layers > 1:
   #    for i in range(lstm_layers - 1):
   #       model.add(
   #             LSTM(
   #                lstm_cells,
   #                input_shape = lstm_input_shape,
   #                return_sequences=True,
   #                dropout=0.0,
   #                recurrent_dropout=0.0
   #             )
   #       )
   #    lstm_input_shape = (2, config.LSTM_CELLS)
   # # Add final (or only) LSTM layer
   # model.add(LSTM(
   #       config.LSTM_CELLS,
   #       input_shape = lstm_input_shape,
   #       return_sequences=False,
   #       dropout=0.0,
   #       recurrent_dropout=0.0
   # ))
   

   # # Add final dense layer
   # model.add(Dense(
   #    units=1,
   #    activation=config.OUTPUT_ACTIVATION,
   #    bias_initializer=keras.initializers.Constant(average_label)
   # ))

   # if config.USE_ADDITIONAL_INPUT:
   #    model.compile(optimizer='adam', loss={'net_profit_head':'mse','additional_input_head':'mse'}, 
   #    loss_weights={'net_profit_head': 0.5, 'additional_input_head': 0.5})
   # else:
   #    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

   return model