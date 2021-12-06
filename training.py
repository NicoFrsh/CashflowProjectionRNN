# main program
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import config
import data_preprocessing
import model

shuffled_validation_split = False

# Read inputs
X_train, y_train, X_test, y_test, scaler_output = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False, include_rfb=False)

# Create validation data 
# TODO: stratified split! Bisher ist ohne shuffle noch besser!
X_val, y_val = [], []
if shuffled_validation_split:
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle= True)

# Input shape = (timesteps, # features)
lstm_input_shape = (config.TIMESTEPS, X_train.shape[2])
print('LSTM input shape:')
print(lstm_input_shape)

# Get average of labels (used as initial bias value)
average_label = np.mean(y_train)
print('average_label = ', average_label)

model_lstm = model.create_rnn_model(lstm_input_shape, average_label)

model_lstm.summary()

# TODO: Install graphviz via brew
# keras.utils.plot_model(model_lstm, 'lstm_model.png', show_shapes = True)

# Create callbacks
callbacks = [ 
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-5 less"
        min_delta=1e-5,
        # "no longer improving" being further defined as "for at least 10 epochs"
        patience=15,
        verbose=1,
    ),
    keras.callbacks.ModelCheckpoint(
        # Path where to save the model 
        # The two parameters below mean that we will overwrite
        # the current checkpoint if and only if
        # the `val_loss` score has improved.
        # The saved model name will include the current epoch.
        filepath='models/mymodel_{0}_{1}.h5'.format(config.LSTM_LAYERS, config.LSTM_CELLS),
        save_weights_only=False,
        save_best_only=True,  # Only save a model if `val_loss` has improved.
        monitor="val_loss",
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1,
    mode='min', min_delta=1e-5, cooldown=0, min_lr=0)
]

# Fit network
history = model_lstm.fit(X_train, y_train, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, validation_split=0.2,
validation_data=(X_val, y_val), verbose=2, callbacks=callbacks, shuffle = True)

print('History metrics:')
print(history.history.keys())

# Plot history of losses
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'])
plt.show()