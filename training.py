# main program
import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import config
import data_preprocessing
import model
from model import RNN_Model

# Set plot font
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
            'font.size' : 11,
            'font.family' : 'lmodern'}
plt.rcParams.update(params)

# Set random seed for reproducibility
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

# Create training and test data
if config.USE_ADDITIONAL_INPUT:
    X_train, y_train, y_2_train, X_val, y_val, y_2_val, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=config.SHUFFLE, train_ratio=config.TRAIN_RATIO)
else:
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=config.SHUFFLE, train_ratio=config.TRAIN_RATIO)


# Input shape = (timesteps, # features)
input_shape = (config.TIMESTEPS + 1, X_train.shape[2])

# Get average of labels (used as initial bias value)
average_label = np.mean(y_train)
average_label_2 = None
if config.USE_ADDITIONAL_INPUT:
    average_label_2 = np.mean(y_2_train)

# model_lstm = model.create_rnn_model(lstm_input_shape, average_label, average_label_2)
# model_lstm = RNN_Model(config.MODEL_TYPE, input_shape, average_label, average_label_2).model
model_lstm = model.build_model(config.MODEL_TYPE, config.USE_ADDITIONAL_INPUT, config.LEARNING_RATE, input_shape,
                                config.LSTM_LAYERS, config.LSTM_CELLS, config.RNN_ACTIVATION, config.OUTPUT_ACTIVATION,
                                config.ADDITIONAL_OUTPUT_ACTIVATION, average_label, average_label_2)

model_lstm.summary()

# Generate descriptive filename for model 
model_name = config.MODEL_PATH + '/model.h5'

## Create callbacks
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
    history = model_lstm.fit(x = X_train, y = [y_train, y_2_train], epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
    validation_data=(X_val, [y_val,y_2_val]), verbose=2, callbacks=callbacks, shuffle = True)

else:
    history = model_lstm.fit(x = X_train, y = y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE,
    validation_data=(X_val, y_val), verbose=2, callbacks=callbacks, shuffle = True)

# Save history object with pickle
import pickle
# Save history object
filename = os.path.dirname(model_name) + '/history.pickle'

with open(filename, 'wb') as f:
    pickle.dump(history.history, f)

# Plot history of losses
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('MSE')
plt.xlabel('Epoche')
plt.legend(['Training', 'Validierung'])
plt.savefig(os.path.dirname(model_name) + '/loss_epochs.pdf')
plt.show()