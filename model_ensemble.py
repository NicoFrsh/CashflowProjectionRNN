import time
import os
import numpy as np
from tensorflow import keras
import pickle
import pandas as pd
import config
import data_preprocessing, data_postprocessing, model

grid_search_path = './grid_search_lstm_gross_surplus/'

results = pd.read_excel(grid_search_path+'results_lstm_gru.xlsx')

number_models = 10
predictions_test, predictions_val = np.zeros((30060, 1)), np.zeros((30000,1))

start_time = time.time()

for i in range(number_models):

    model_type = results['model_type'][i]

    # Extract parameters from model name
    parameters = results['model_name'][i].split('_')    
    timesteps = int(parameters[2])
    batch_size = int(parameters[4])
    recurrent_activation = parameters[5]
    y_activation = parameters[6]
    z_activation = parameters[7]
    rnn_layers = int(parameters[8])
    rnn_cells = int(parameters[9])

    X_train, y_train, y_2_train, X_val, y_val, y_2_val, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, projection_time=config.PROJECTION_TIME, 
    recurrent_timesteps=timesteps, output_activation=y_activation, additional_output_activation=z_activation, shuffle_data=config.SHUFFLE,
    train_ratio=config.TRAIN_RATIO)

    # Input shape = (timesteps+1, # features)
    input_shape = (timesteps + 1, X_train.shape[2])

    # Get average of labels (used as initial bias value)
    average_label = np.mean(y_train)
    average_label_2 = None
    if config.USE_ADDITIONAL_INPUT:
        average_label_2 = np.mean(y_2_train)

    rnn_model = model.build_model(model_type, config.USE_ADDITIONAL_INPUT, config.LEARNING_RATE, input_shape,
                            rnn_layers, rnn_cells, recurrent_activation, y_activation, z_activation,
                            average_label, average_label_2)

    # Load trained model
    if model_type == 'lstm':
        rnn_model.load_weights(grid_search_path + results['model_name'][i] + '/model.h5')

    else:
        rnn_model.load_weights('./grid_search_gru_gross_surplus/' + results['model_name'][i] + '/model.h5')

    rnn_model.summary()
    # Make predictions
    pred_test, _ = data_postprocessing.recursive_prediction(X_test, rnn_model, timesteps, batch_size)
    pred_val, _ = data_postprocessing.recursive_prediction(X_val, rnn_model, timesteps, batch_size)

    # Sum up predictions
    predictions_test += pred_test
    predictions_val += pred_val


# Divide by number of models in ensemble to get mean
predictions_test = predictions_test / number_models
predictions_val = predictions_val / number_models

print(f'Execution time: {time.time() - start_time} seconds')

# Obtain original scaled data
pred_test_original = scaler_output.inverse_transform(predictions_test)
y_test_original = scaler_output.inverse_transform(y_test)

pred_val_original = scaler_output.inverse_transform(predictions_val)
y_val_original = scaler_output.inverse_transform(y_val)

# Save using pickle
ensemble_path = grid_search_path + f'Ensemble_{number_models}'
os.makedirs(ensemble_path, exist_ok=True)
with open(ensemble_path + '/data.pickle' , 'wb') as f:
    pickle.dump([predictions_test, y_test, pred_test_original, y_test_original, predictions_val, y_val, pred_val_original, y_val_original], f)