# Script to save predictions using pickle
import pickle
from tensorflow import keras
import numpy as np

import config
import data_preprocessing
import model
import data_postprocessing

if config.USE_ADDITIONAL_INPUT:
    model_path = 'models/model_acc_{}_1_32.h5'.format(str.replace(config.ADDITIONAL_INPUT, ' ', '_'))
else:
    model_path = 'models/sgd_model_acc_{0}_{1}.h5'.format(config.LSTM_LAYERS, config.LSTM_CELLS)

# Create training and test data
if config.USE_ADDITIONAL_INPUT:
    X_train, y_train, y_2_train, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)
else:
    X_train, y_train, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)

# Input shape = (timesteps, # features)
lstm_input_shape = (config.TIMESTEPS, X_train.shape[2])

# Get average of labels (used as initial bias value)
average_label = np.mean(y_train)
average_label_2 = None
if config.USE_ADDITIONAL_INPUT:
    average_label_2 = np.mean(y_2_train)

model_lstm = model.create_lstm_model(lstm_input_shape, average_label, average_label_2)

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

# Save predictions array
with open('./models/data.pickle', 'wb') as f:
    pickle.dump([predictions_np,y_test], f)