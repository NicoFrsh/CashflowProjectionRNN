from keras.engine.base_layer import Layer
from tensorflow import keras
from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, BatchNormalization, Dropout, Embedding
import config

class RNN_Model:

   def __init__(self, model_type, input_shape, average_label, average_label_2):
       
      self.model = self._build_model(model_type, input_shape, config.LSTM_LAYERS, config.LSTM_CELLS, average_label, average_label_2)

   def _build_model(self, model_type, input_shape, recurrent_layers, recurrent_cells, average_label, average_label_2 = None):
      # Parameters:
      #     - model_type (string):                 'simple_rnn', 'lstm' or 'gru'
      #     - input_shape (...):    
      #     - recurrent_layers (int):              # of recurrent layers 
      #     - recurrent_cells (int/list(int)):     # of cells in recurrent layer(s). If more than 1 recurrent layer is used, this parameter is a list of integers.
      #     - average_label (float):               Average of the target values (labels). Used to initialize the bias parameter of the output layer for better convergence.

      input = Input(shape=input_shape)

      current_output = input

      # TODO: NOT WORKING!!
      if recurrent_layers > 1:
         for i in range(recurrent_layers - 1):

            if model_type == 'simple_rnn':
               current_output = SimpleRNN(int(recurrent_cells / (2**i)), activation=config.RNN_ACTIVATION, return_sequences=True, dropout=0.1)(current_output)
            
            elif model_type == 'lstm':
               current_output = LSTM(int(recurrent_cells / (2**i)), activation=config.RNN_ACTIVATION, return_sequences=True, dropout=0.1)(current_output)

            elif model_type == 'gru':
               current_output = GRU(int(recurrent_cells / (2**i)), activation=config.RNN_ACTIVATION, return_sequences=True, dropout=0.1)(current_output)

      if model_type == 'simple_rnn':
         current_output = SimpleRNN(int(recurrent_cells / (2**(recurrent_layers-1))), activation=config.RNN_ACTIVATION)(current_output)
      
      elif model_type == 'lstm':
         current_output = LSTM(int(recurrent_cells / (2**(recurrent_layers-1))), activation=config.RNN_ACTIVATION)(current_output)

      elif model_type == 'gru':
         current_output = GRU(int(recurrent_cells / (2**(recurrent_layers-1))), activation=config.RNN_ACTIVATION)(current_output)

      # Add Batch Normalization
      # current_output = BatchNormalization()(current_output)

      # Add Dropout Layer (performs worse)
      # current_output = Dropout(0.2)(current_output)

      net_profit_head = Dense(1, activation=config.OUTPUT_ACTIVATION, bias_initializer=keras.initializers.Constant(average_label),
      name = 'net_profit_head')(current_output)

      if config.USE_ADDITIONAL_INPUT: # Additional network head for additional input
         additional_output_head = Dense(1, activation=config.ADDITIONAL_OUTPUT_ACTIVATION, bias_initializer=keras.initializers.Constant(average_label_2),
         name = 'additional_input_head')(current_output)
      
         model = keras.Model(inputs = input, outputs =[net_profit_head, additional_output_head])

      else:
         model = keras.Model(inputs = input, outputs = net_profit_head)

      if config.USE_ADDITIONAL_INPUT:
         model.compile(optimizer='adam', loss={'net_profit_head':'mse','additional_input_head':'mse'}, metrics={'net_profit_head':'mae','additional_input_head':'mae'},
         loss_weights={'net_profit_head': 0.5, 'additional_input_head': 0.5})
      else:
         # model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.005), loss='mse', metrics=['mae'])
         model.compile(optimizer='adam', loss='mse', metrics=['mae'])

      return model

   def predict(self, x):
      return self.model.predict(x)

   def predict_recursively(self, x):
      # TODO: Implement!
      return 0
