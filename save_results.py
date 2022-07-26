# Script to save predictions using pickle
import os
import pickle
from tensorflow import keras
import numpy as np

import config
import data_preprocessing
import model
from model import RNN_Model
import data_postprocessing

# model_path = config.MODEL_PATH + '/model.h5'
path = 'grid_search_lstm_gross_surplus/Ensemble_5'
model_path = path + '/model.h5'
# Fixed parameters
# model_type = 'lstm'
# use_additional_input = False
# yearly = True
# discounted = True
# path = f'new_grid_search_{model_type}_add_input_tanh_rnn_tanh/'
# path += '9_T_10_BS_250_tanh_linear_tanh_1_64'
# model_path = path + '/model.h5'

print('Model path: ', model_path)

# Create training and test data
if config.USE_ADDITIONAL_INPUT:
    X_train, y_train, y_2_train, X_val, y_val, y_2_val, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=config.SHUFFLE, train_ratio=config.TRAIN_RATIO)
else:
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=config.SHUFFLE, train_ratio=config.TRAIN_RATIO)

# Input shape = (timesteps+1, # features)
input_shape = (config.TIMESTEPS + 1, X_train.shape[2])

# Get average of labels (used as initial bias value)
average_label = np.mean(y_train)
average_label_2 = None
if config.USE_ADDITIONAL_INPUT:
    average_label_2 = np.mean(y_2_train)

# model_lstm = RNN_Model(config.MODEL_TYPE, input_shape, average_label, average_label_2).model
model_lstm = model.build_model(config.MODEL_TYPE, config.USE_ADDITIONAL_INPUT, config.LEARNING_RATE, input_shape,
                        config.LSTM_LAYERS, config.LSTM_CELLS, config.RNN_ACTIVATION, config.OUTPUT_ACTIVATION, config.ADDITIONAL_OUTPUT_ACTIVATION,
                        average_label, average_label_2)

model_lstm.summary()

# Load trained model
model_lstm.load_weights(model_path)

# Make predictions
# If an additional input, e.g. RfB, is used the net shall predict the test data using the predicted additional inputs recursively.
if config.USE_ADDITIONAL_INPUT:
    pred_test, _ = data_postprocessing.recursive_prediction(X_test, model_lstm, config.TIMESTEPS, config.BATCH_SIZE)
    # pred_test, pred_add_output_test = model_lstm.predict(X_test)
    pred_train, _ = model_lstm.predict(X_train)
    # pred_val, _ = model_lstm.predict(X_val)
    pred_val, _ = data_postprocessing.recursive_prediction(X_val, model_lstm, config.TIMESTEPS, config.BATCH_SIZE)
else:
    pred_test = model_lstm.predict(X_test)
    pred_train = model_lstm.predict(X_train)
    pred_val = model_lstm.predict(X_val)

# Obtain original scaled data
pred_test_original = scaler_output.inverse_transform(pred_test)
y_test_original = scaler_output.inverse_transform(y_test)

pred_train_original = scaler_output.inverse_transform(pred_train)
y_train_original = scaler_output.inverse_transform(y_train)

pred_val_original = scaler_output.inverse_transform(pred_val)
y_val_original = scaler_output.inverse_transform(y_val)

data_dict = {
    "pred_train" : pred_train,
    "y_train" : y_train,
    "pred_train_original" : pred_train_original,
    "y_train_original" : y_train_original,

    "pred_val" : pred_val,
    "y_val" : y_val,
    "pred_val_original" : pred_val_original,
    "y_val_original" : y_val_original,

    "pred_test" : pred_test,
    "y_test" : y_test,
    "pred_test_original" : pred_test_original,
    "y_test_original" : y_test_original,
}

# Save predictions array
filename = config.MODEL_PATH + '/data.pickle'
# filename = path + '/data.pickle'
# filename = path + '/data.pickle'

with open(filename, 'wb') as f:
    pickle.dump(data_dict, f)
    # pickle.dump(
        # [pred_test,y_test, pred_test_original, y_test_original, pred_val, y_val, pred_val_original, y_val_original], f)