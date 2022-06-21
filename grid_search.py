# Hyperparameter search using keras tuner
import numpy as np
from tensorflow import keras
import keras_tuner

import config
import data_preprocessing
from model import RNN_Model

# Create training, validation and test data
if config.USE_ADDITIONAL_INPUT:
    X_train, y_train, y_2_train, X_val, y_val, y_2_val, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False, train_ratio=config.TRAIN_RATIO)
else:
    X_train, y_train, X_val, y_val, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False, train_ratio=config.TRAIN_RATIO)

# Input shape = (timesteps, # features)
input_shape = (config.TIMESTEPS + 1, X_train.shape[2])

# Get average of labels (used as initial bias value)
average_label = np.mean(y_train)
average_label_2 = None
if config.USE_ADDITIONAL_INPUT:
    average_label_2 = np.mean(y_2_train)

rnn_model = RNN_Model(model_type = 'lstm', input_shape = input_shape, average_label=average_label, average_label_2=average_label_2)

def build_model(hp):
    # model_type = hp.Choice('model_type', ['simple_rnn', 'lstm', 'gru'])
    # lr = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling="log")

    model = rnn_model._build_model(model_type = config.MODEL_TYPE, input_shape = rnn_model.input_shape, recurrent_layers = config.LSTM_LAYERS, 
    recurrent_cells = config.LSTM_CELLS, average_label = rnn_model.average_label, average_label_2 = rnn_model.average_label_2)

    return model

# build_model(keras_tuner.HyperParameters(), rnn_model)

tuner = keras_tuner.RandomSearch(hypermodel=build_model, objective='val_loss', max_trials= 3, executions_per_trial = 1, overwrite = True,
directory = 'grid_search', project_name = 'model_types')

tuner.search_space_summary()

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
    # keras.callbacks.ModelCheckpoint(
    #     # Path where to save the model 
    #     # The two parameters below mean that we will overwrite
    #     # the current checkpoint if and only if
    #     # the `val_loss` score has improved.
    #     # The saved model name will include the current epoch.
    #     filepath=model_name,
    #     save_weights_only=False,
    #     save_best_only=True,  # Only save a model if `val_loss` has improved.
    #     monitor="val_loss",
    #     verbose=1,
    # ),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1,
    mode='min', min_delta=1e-5, cooldown=0, min_lr=0)
]

# tuner.search(X_train, y_train, batch_size = config.BATCH_SIZE, epochs = config.EPOCHS, callbacks = callbacks, validation_data = (X_val, y_val))

best_model = tuner.get_best_models(num_models=1)

print('best_model: ', best_model)

best_model.build()
best_model.summary()
