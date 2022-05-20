# main program
import os
import random
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import config
import data_preprocessing
from model import RNN_Model

# Set random seed for reproducibility
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

shuffled_validation_split = False

# Create training and test data
if config.USE_ADDITIONAL_INPUT:
    X_train, y_train, y_2_train, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)
    # Create validation data 
    X_train, y_train, y_2_train, X_val, y_val, y_2_val = data_preprocessing.train_test_split(X_train, y_train, y_2_train, 0.8)
else:
    X_train, y_train, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)
    # Create validation data
    X_train, y_train, X_val, y_val = data_preprocessing.train_test_split(X_train, y_train, [], 0.8)


# TODO: stratified split! Bisher ist ohne shuffle noch besser!
# TODO: x_2_val und y_2_val fuer additional input
if shuffled_validation_split:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle= True)

# Input shape = (timesteps, # features)
input_shape = (config.TIMESTEPS + 1, X_train.shape[2])

# Get average of labels (used as initial bias value)
average_label = np.mean(y_train)
average_label_2 = None
if config.USE_ADDITIONAL_INPUT:
    average_label_2 = np.mean(y_2_train)

# model_lstm = model.create_rnn_model(lstm_input_shape, average_label, average_label_2)
model_lstm = RNN_Model(config.MODEL_TYPE, input_shape, average_label, average_label_2).model

model_lstm.summary()

# TODO: Install graphviz via brew
# keras.utils.plot_model(model_lstm, 'lstm_model.png', show_shapes = True)

## Create callbacks
# Generate descriptive filename for model 
model_name = config.MODEL_PATH + '/model.h5'
# model_name = 'models/new_'
# if config.use_yearly_inputs:
#     model_name += 'yearly_'
# if config.use_discounted_np:
#     model_name += 'discounted_'
# if config.RNN_ACTIVATION != 'tanh':
#     model_name += config.RNN_ACTIVATION + '_' + config.OUTPUT_ACTIVATION + '_'
# model_name += 'T_{0}_bs_{1}_{2}_'.format(config.TIMESTEPS, config.BATCH_SIZE, config.MODEL_TYPE)
# if config.USE_ADDITIONAL_INPUT:
#     additional_input_str = str.replace(config.ADDITIONAL_INPUT, " ", "_")
#     model_name += '{0}_'.format(additional_input_str)
# model_name += '{0}_{1}/model.h5'.format(config.LSTM_LAYERS, config.LSTM_CELLS)

os.makedirs(os.path.dirname(model_name), exist_ok=True)

callbacks = [ 
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-5 less"
        min_delta=1e-5,
        # "no longer improving" being further defined as "for at least 20 epochs"
        patience=20,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        # Path where to save the model 
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath=model_name,
        save_weights_only=False,
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,
    mode='min', min_delta=1e-5, cooldown=0, min_lr=0)
]

# Fit network
if config.USE_ADDITIONAL_INPUT:
    history = model_lstm.fit(x = X_train, y = [y_train, y_2_train], epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_split=0.2,
    validation_data=(X_val, y_val), verbose=2, callbacks=callbacks, shuffle = True)

else:
    history = model_lstm.fit(x = X_train, y = y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_split=0.2,
    validation_data=(X_val, y_val), verbose=2, callbacks=callbacks, shuffle = True)

# Save history object with pickle
import pickle
# Save predictions array
filename = os.path.dirname(model_name) + '/history.pickle'

with open(filename, 'wb') as f:
    pickle.dump(history.history, f)

# Plot history of losses
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.savefig(os.path.dirname(model_name) + '/loss_epochs.png')
plt.show()