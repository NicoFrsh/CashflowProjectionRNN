from tensorflow import keras
import tensorflow as tf
from keras.layers import Input, SimpleRNN, LSTM, GRU, Dense
import config

def build_model(model_type, use_additional_input, learning_rate, input_shape, recurrent_layers, recurrent_cells,
               recurrent_activation, output_activation, additional_output_activation, average_label, average_label_2 = None):
   # Parameters:
   #     - model_type (string):                 'simple_rnn', 'lstm' or 'gru'
   #     - input_shape (...):    
   #     - recurrent_layers (int):              # of recurrent layers 
   #     - recurrent_cells (int/list(int)):     # of cells in recurrent layer(s). If more than 1 recurrent layer is used, this parameter is a list of integers.
   #     - average_label (float):               Average of the target values (labels). Used to initialize the bias parameter of the output layer for better convergence.

   input = Input(shape=input_shape)

   current_output = input

   dropout = config.DROPOUT_RATE

   if recurrent_layers > 1:
      for i in range(recurrent_layers - 1):

         if model_type == 'simple_rnn':
            current_output = SimpleRNN(int(recurrent_cells / (2**i)), activation=recurrent_activation, return_sequences=True, dropout=dropout)(current_output)
         
         elif model_type == 'lstm':
            current_output = LSTM(int(recurrent_cells / (2**i)), activation=recurrent_activation, return_sequences=True, dropout=dropout)(current_output)
            # current_output = LSTM(recurrent_cells, activation='linear', return_sequences=True, dropout=dropout)(current_output)

         elif model_type == 'gru':
            current_output = GRU(int(recurrent_cells / (2**i)), activation=recurrent_activation, return_sequences=True, dropout=dropout)(current_output)

   if model_type == 'simple_rnn':
      current_output = SimpleRNN(int(recurrent_cells / (2**(recurrent_layers-1))), activation=recurrent_activation, dropout=dropout)(current_output)
   
   elif model_type == 'lstm':
      current_output = LSTM(int(recurrent_cells / (2**(recurrent_layers-1))), activation=recurrent_activation, dropout=dropout)(current_output)
      # current_output = LSTM(recurrent_cells, activation=recurrent_activation, dropout=dropout)(current_output)

   elif model_type == 'gru':
      current_output = GRU(int(recurrent_cells / (2**(recurrent_layers-1))), activation=recurrent_activation, dropout=dropout)(current_output)

   # Add Batch Normalization
   # current_output = BatchNormalization()(current_output)
   # current_output = LayerNormalization()(current_output)

   # Add Dropout Layer (performs worse)
   # if dropout > 0.0:
   #    current_output = Dropout(dropout)(current_output)

   # TODO: Ausprobieren: Zusaetzlicher Dense Layer mit mehr Neuronen und vielleicht ReLu vor Output Dense Layer

   # current_output = Dense(recurrent_cells, activation='relu')(current_output)

   net_profit_head = Dense(1, activation=output_activation, bias_initializer=keras.initializers.Constant(average_label),
   name = 'net_profit_head')(current_output)

   if use_additional_input: # Additional network head for additional input
      additional_output_head = Dense(1, activation=additional_output_activation, bias_initializer=keras.initializers.Constant(average_label_2),
      name = 'additional_input_head')(current_output)
   
      model = keras.Model(inputs = input, outputs =[net_profit_head, additional_output_head])

   else:
      model = keras.Model(inputs = input, outputs = net_profit_head)

   if use_additional_input:
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss={'net_profit_head':'mse','additional_input_head':'mse'}, metrics={'net_profit_head':'mae','additional_input_head':'mae'},
      loss_weights={'net_profit_head': 0.5, 'additional_input_head': 0.5})
   else:
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])

   return model