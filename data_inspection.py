# Data inspection
# Evaluate saved model
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
    X_train, y_train, y_2_train, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)
else:
    X_train, y_train, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)

print('---------------')
print('Shape X_test: ', X_test.shape)
X_train_extract = X_train[1,:,:]
print('Shape of inputs extract: ', X_train_extract[:,:-1].shape)
print('Shape of additional_inputs extract: ', X_train_extract[:,-1].shape)
print('Shape of additional_inputs extract after reshape: ', np.reshape(X_train_extract[:,-1], (2,1)).shape)
print('Inverse transform of additional_input: ', scaler_additional_input.inverse_transform(np.reshape(X_train_extract[:,-1], (2,1))))

# X_train_extract = np.array( (scaler_input.inverse_transform(X_train_extract[:,:-1]), scaler_additional_input.inverse_transform(np.reshape(X_train_extract[:,-1], (2,1)))) )
# print(X_train_extract)
# print('__________________ reshaping ---------------')
# X_train_extract = np.reshape(X_train_extract, (-1,2,X_train.shape[2]))
# print('Shape extract: ', X_train_extract.shape)
# print(X_train_extract)