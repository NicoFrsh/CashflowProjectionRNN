# Script to save predictions using pickle
import os
import pickle
from tensorflow import keras
import numpy as np

import config
import data_preprocessing
from model import RNN_Model
import data_postprocessing

if config.USE_ADDITIONAL_INPUT:
    model_path = 'models/bs_{0}_{1}_{2}_{3}_{4}/model.h5'.format(config.BATCH_SIZE, config.MODEL_TYPE, str.replace(config.ADDITIONAL_INPUT, ' ', '_'), config.LSTM_LAYERS, config.LSTM_CELLS)
else:
    model_path = 'models/test_bs_{0}_{1}_{2}_{3}/model.h5'.format(config.BATCH_SIZE, config.MODEL_TYPE, config.LSTM_LAYERS, config.LSTM_CELLS)

# Create training and test data
if config.USE_ADDITIONAL_INPUT:
    X_train, y_train, y_2_train, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)
else:
    X_train, y_train, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)

# Input shape = (timesteps, # features)
input_shape = (config.TIMESTEPS, X_train.shape[2])

# Get average of labels (used as initial bias value)
average_label = np.mean(y_train)
average_label_2 = None
if config.USE_ADDITIONAL_INPUT:
    average_label_2 = np.mean(y_2_train)

model_lstm = RNN_Model(config.MODEL_TYPE, input_shape, average_label, average_label_2).model

model_lstm.summary()

# Load trained model
model_lstm.load_weights(model_path)

# Make predictions
# If an additional input, e.g. RfB, is used the net shall predict the test data using the predicted additional inputs recursively.
if config.USE_ADDITIONAL_INPUT:
    predictions_np, predictions_additional_input = data_postprocessing.recursive_prediction(X_test, model_lstm)
    # predictions_np, predictions_additional_input = model_lstm.predict(X_test)
    predictions_np_train, predictions_additional_input_train = model_lstm.predict(X_train)
else:
    # predictions_np = data_postprocessing.recursive_prediction(X_test, model_lstm)
    predictions_np = model_lstm.predict(X_test)
    predictions_np_train = model_lstm.predict(X_train)

# Obtain original scaled data
predictions_original = scaler_output.inverse_transform(predictions_np)
y_original = scaler_output.inverse_transform(y_test)

# Save predictions array
filename = os.path.dirname(model_path) + '/data.pickle'

with open(filename, 'wb') as f:
    pickle.dump([predictions_np,y_test, predictions_original, y_original, predictions_np_train, y_train], f)