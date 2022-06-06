# Data inspection
# Evaluate saved model
from statistics import mean
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

import config
import data_preprocessing
import model
import data_postprocessing

# Read inputs
if config.USE_ADDITIONAL_INPUT:
    X_train, y_train, y_2_train, X_val, y_val, y_2_val, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=config.SHUFFLE)
else:
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, projection_time=config.PROJECTION_TIME,
    recurrent_timesteps= config.TIMESTEPS, shuffle_data=config.SHUFFLE)

print('X_train shape: ', X_train.shape)

labels = scaler_output.inverse_transform(y_train)
print('first label: ')
print(labels[0])
features_extract = X_train[0, :, :]
# features = [scaler_input.inverse_transform(X_train[:,i,:]) for i in range(config.TIMESTEPS)]
# features = np.array(features)
print('shape of features: ', features_extract.shape)
# features = scaler_input.inverse_transform(X_train.reshape())

print('first features: ')   
print(features_extract)

# print('---------------')
# print('Shape X_test: ', X_test.shape)
# X_train_extract = X_train[1,:,:]
# print('Shape of inputs extract: ', X_train_extract[:,:-1].shape)
# print('Shape of additional_inputs extract: ', X_train_extract[:,-1].shape)
# print('Shape of additional_inputs extract after reshape: ', np.reshape(X_train_extract[:,-1], (2,1)).shape)
# print('Inverse transform of additional_input: ', scaler_additional_input.inverse_transform(np.reshape(X_train_extract[:,-1], (2,1))))

# Get min, max and mean of additional_input
min_train, max_train, mean_train = np.min(scaler_additional_input.inverse_transform(y_2_train)), np.max(scaler_additional_input.inverse_transform(y_2_train)), np.mean(scaler_additional_input.inverse_transform(y_2_train))
min_test, max_test, mean_test = np.min(scaler_additional_input.inverse_transform(y_2_test)), np.max(scaler_additional_input.inverse_transform(y_2_test)), np.mean(scaler_additional_input.inverse_transform(y_2_test))

print('Additional input is set to {}'.format(config.ADDITIONAL_INPUT))
print('TRAIN: Min: {}, Max: {}, Mean: {}'.format(min_train, max_train, mean_train))
print('TEST: Min: {}, Max: {}, Mean: {}'.format(min_test, max_test, mean_test))


# X_train_extract = np.array( (scaler_input.inverse_transform(X_train_extract[:,:-1]), scaler_additional_input.inverse_transform(np.reshape(X_train_extract[:,-1], (2,1)))) )
# print(X_train_extract)
# print('__________________ reshaping ---------------')
# X_train_extract = np.reshape(X_train_extract, (-1,2,X_train.shape[2]))
# print('Shape extract: ', X_train_extract.shape)
# print(X_train_extract)