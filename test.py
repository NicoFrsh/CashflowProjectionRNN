# TEST
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import config
import data_preprocessing, data_postprocessing, model

model_path = 'models/model_1_32.h5'

# Create training and test data
X_train, y_train, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)


net_profit_original = scaler_output.inverse_transform(y_train)

# additional_input_original = scaler_input.inverse_transform(y_train[:,1])
# print('y_train head: ', net_profit_original[:10,:], additional_input_original[:10,:])
# TEST
# y_hat = np.zeros_like(y_test)
# y_hat[1::59,0] = 1
# y_hat[2::59,0] = 2
# feature = X_test[2::59,:,:]
# print('first feature before: ', feature[0,:,:])
# print('second feature before: ', feature[1,:,:])
# feature[:,1,-1] = y_hat[1::59,0]
# feature[:,0,-1] = y_hat[0::59, 0]
# print('first feature after: ', feature[0,:,:])
# print('second feature after: ', feature[1,:,:])


# Input shape = (timesteps, # features)
lstm_input_shape = (config.TIMESTEPS, X_train.shape[2])

rnn_model = model.create_rnn_model(lstm_input_shape, 0.5)

print(rnn_model.summary())

rnn_model.load_weights(model_path)

print('OLD RECURSIVE_PREDICTION:')
predictions_old = data_postprocessing.recursive_prediction_old(X_test, rnn_model)
print('RECURSIVE_PREDICTION:')
predictions = data_postprocessing.recursive_prediction(X_test, rnn_model)

t0 = predictions[0,0]
t0_old = predictions_old[0,0]

t1 = predictions[1,0]
t1_old = predictions_old[1,0]
# t1 = predictions[60,0]
# t1_old = predictions_old[60,0]

print("{:.32f}".format(t0))
print("{:.32f}".format(t0_old))

print("{:.32f}".format(t1))
print("{:.32f}".format(t1_old))

print(t0 - t0_old)
print(t1 - t1_old)

print(t0 == t0_old)
print(t1 == t1_old)

print(type(t0))
print(type(t0_old))
