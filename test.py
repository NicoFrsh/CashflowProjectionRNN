# TEST
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import config
import data_preprocessing, data_postprocessing, model

model_path = 'models/mymodel_1_32_without_rfb.h5'

# Create training and test data
X_train, y_train, X_test, y_test, scaler_output = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False, include_rfb=config.USE_ADDITIONAL_INPUT)

# TEST
y_hat = np.zeros_like(y_test)
y_hat[1::59,0] = 1
y_hat[2::59,0] = 2
feature = X_test[2::59,:,:]
print('first feature before: ', feature[0,:,:])
print('second feature before: ', feature[1,:,:])
feature[:,1,-1] = y_hat[1::59,0]
feature[:,0,-1] = y_hat[0::59, 0]
print('first feature after: ', feature[0,:,:])
print('second feature after: ', feature[1,:,:])


# # Input shape = (timesteps, # features)
# lstm_input_shape = (config.TIMESTEPS, X_train.shape[2])

# rnn_model = model.create_rnn_model(lstm_input_shape, 0.5)

# rnn_model.load_weights(model_path)

# predictions = data_postprocessing.recursive_prediction(X_test, rnn_model)
# print('shape of recursive predictions: ', predictions.shape)


# y_hat = np.empty((1, 59059))
# print('Shape y_hat: ', y_hat.shape)

# for i in range(59):
        
#         if i == 0: # (t = 1): Take actual net profit from timestep 0 for both input vectors (padding!)
#             y_hat_i = rnn_model.predict(np.reshape(X_test[i::59,:,:], (-1,2,num_features)))

#         elif i == 1: # (i.e. t = 2): Take actual net profit from timestep 0 for the first input vector
#             feature = X_test[i::59, :, :]
#             feature[:,1,-1] = y_hat[i-1::59]
#             y_hat_i = rnn_model.predict(np.reshape(feature, (-1,2,num_features)))
#         else:
#             # implement!
#             feature = X_test[i::59, :, :]

#         print('shape y_hat_i: ', y_hat_i.shape)
#         y_hat[i::59] = y_hat_i
