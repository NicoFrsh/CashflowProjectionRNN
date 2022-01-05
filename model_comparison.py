# Script to compare performance of two or more models
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import config
import data_preprocessing
import model
import data_postprocessing

# Which models do you want to compare?
model_1_path = 'models/mymodel_1_32_without_rfb.h5'
model_2_path = 'models/mymodel_1_32_rfb.h5'

# Read inputs
X_train_1, y_train_1, X_test_1, y_test_1, scaler_output = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)

X_train_2, y_train_2, X_test_2, y_test_2, scaler_output = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False, include_rfb=True)
# X_train_2, y_train_2, X_test_2, y_test_2, scaler_output = X_train_1, y_train_1, X_test_1, y_test_1, scaler_output

# Input shape = (timesteps, # features)
lstm_input_shape_1 = (config.TIMESTEPS, X_train_1.shape[2])
lstm_input_shape_2 = (config.TIMESTEPS, X_train_2.shape[2])

# Get average of labels (used as initial bias value)
average_label = np.mean(y_train_1)

model_1 = model.create_rnn_model(lstm_input_shape_1, average_label)
model_2 = model.create_rnn_model(lstm_input_shape_2, average_label)

model_1.load_weights(model_1_path)
model_2.load_weights(model_2_path)

# Evaluate networks
# Get test accuracy
# score_1 = model_1.evaluate(X_test_1, y_test_1, verbose=0)
# print(f'Model 1: \r\nTest loss: {score_1[0]} / Test mae: {score_1[1]}')
# score_2 = model_2.evaluate(X_test_2, y_test_2, verbose=0)
# print(f'Model 2: \r\nTest loss: {score_2[0]} / Test mae: {score_2[1]}')

# # Get training accuracy
# score_1 = model_1.evaluate(X_train_1, y_train_1, verbose=0)
# print(f'Model 1: \r\nTrain loss: {score_1[0]} / Train mae: {score_1[1]}')
# score_2 = model_2.evaluate(X_train_2, y_train_2, verbose=0)
# print(f'Model 2: \r\nTrain loss: {score_2[0]} / Train mae: {score_2[1]}')

# Make predictions
predictions_test_1 = data_postprocessing.recursive_prediction(X_test_1, model_1)
# predictions_train_1 = model_1.predict(X_train_1)

predictions_test_2 = data_postprocessing.recursive_prediction(X_test_2, model_2, additional_input=True)
# predictions_train_2 = model_2.predict(X_train_2)

# Compute and plot training MSE over timesteps
# train_loss_1 = data_postprocessing.calculate_loss_per_timestep(y_train_1, predictions_train_1)
# train_loss_2 = data_postprocessing.calculate_loss_per_timestep(y_train_2, predictions_train_2)

x = range(1,60)

# plt.figure()
# plt.plot(x, train_loss_1, label = 'without {}'.format(config.ADDITIONAL_INPUT))
# plt.plot(x, train_loss_2, label = 'with {}'.format(config.ADDITIONAL_INPUT))
# plt.legend()
# plt.xlabel('year')
# plt.ylabel('MSE')
# plt.title('Train MSE over time')

test_loss_1 = data_postprocessing.calculate_loss_per_timestep(y_test_1, predictions_test_1)
test_loss_2 = data_postprocessing.calculate_loss_per_timestep(y_test_2, predictions_test_2)

plt.figure()
plt.plot(x, test_loss_1, label = 'without {}'.format(config.ADDITIONAL_INPUT))
plt.plot(x, test_loss_2, label = 'with {}'.format(config.ADDITIONAL_INPUT))
plt.legend()
plt.xlabel('year')
plt.ylabel('MSE')
plt.title('Test MSE over time')

plt.show()