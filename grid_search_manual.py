# Manual grid search
from tensorflow import keras
import numpy as np
import tensorflow as tf
import random
import os
import matplotlib.pyplot as plt
import pickle
import csv
import config
from model import build_model
import data_preprocessing
import data_postprocessing

# Set random seed to get reproducible combinations
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)
tf.random.set_seed(config.RANDOM_SEED)

# Grid search configurations
sample_ratio = 0.15

# Fixed parameters
model_type = 'lstm'
use_additional_input = False
learning_rate = 0.005
yearly = True
discounted = True
path = f'test_shuffle_stability_{model_type}_lr_{str.replace(str(learning_rate), ".", "")}_yearly_discounted/'

print('PATH: ', path)

# Define hyperparameter grid
shuffle = [True]#, False]
timesteps = [2]#,4,6,8]
lstm_cells = [32]#, 64, 128]
lstm_layers = [1]#, 2, 3, 4]
use_dropout = [False]#,True]
batch_size = [100, 250, 500, 1000]
rnn_activation = ['tanh']#, 'relu', 'linear']
output_activation = ['tanh']#, 'linear']
additional_output_activation = ['tanh']#, 'linear', 'sigmoid']

number_combinations = len(shuffle)*len(timesteps)*len(lstm_cells)*len(lstm_layers)*len(use_dropout)*len(batch_size)*len(rnn_activation)*len(output_activation)*len(additional_output_activation)
print('total # of combinations: ', number_combinations)

# sample combinations
number_sample = int(number_combinations * sample_ratio)
print(f'sampling {number_sample} combinations.')

results = []

for i in range(2):

    random_seed = i

    tf.keras.backend.clear_session()

    shuffle_i = random.choice(shuffle)
    timesteps_i = random.choice(timesteps)
    lstm_cells_i = random.choice(lstm_cells)
    lstm_layers_i = random.choice(lstm_layers)
    if lstm_layers_i > 1: # multilayer rnn need less cells
        lstm_cells_i = int(lstm_cells_i / 2)
    # use_dropout_i = random.choice(use_dropout)
    batch_size_i = random.choice(batch_size)
    rnn_activation_i = random.choice(rnn_activation)
    output_activation_i = random.choice(output_activation)
    additional_output_activation_i = random.choice(additional_output_activation)

    model_name = f'{i}_'
    if shuffle_i:
        model_name += 'shuffle_'
    model_name += f'T_{timesteps_i}_BS_{batch_size_i}_{rnn_activation_i}_{output_activation_i}_{lstm_layers_i}_{lstm_cells_i}'
    model_dict = dict()
    for element in results:
        if element['model_name'] == model_name:
            continue

    model_dict['model_name'] = model_name

    model_path = path + model_name

    # Prepare data
    # Create training and test data
    if use_additional_input:
        X_train, y_train, y_2_train, X_val, y_val, y_2_val, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
        config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, recurrent_timesteps=timesteps_i, output_activation=output_activation_i,
        additional_output_activation=additional_output_activation_i, shuffle_data=shuffle_i, train_ratio=config.TRAIN_RATIO)
    else:
        X_train, y_train, X_val, y_val, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
        config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, recurrent_timesteps=timesteps_i, output_activation=output_activation_i,
        additional_output_activation=additional_output_activation_i, shuffle_data=shuffle_i, train_ratio=config.TRAIN_RATIO)

    # Get input shape
    input_shape = (timesteps_i + 1, X_train.shape[2])

    # Get average of labels (used as initial bias value)
    average_label = np.mean(y_train)
    average_label_2 = None
    if use_additional_input:
        average_label_2 = np.mean(y_2_train)


    # Build model
    model = build_model(model_type, use_additional_input, learning_rate, input_shape, lstm_layers_i, lstm_cells_i, rnn_activation_i, 
    output_activation_i, additional_output_activation_i, average_label, average_label_2)

    model.summary()

    # Create callbacks
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
        filepath= model_path + '/model.h5',
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
        history = model.fit(x = X_train, y = [y_train, y_2_train], epochs=config.EPOCHS, batch_size=batch_size_i,
        validation_data=(X_val, y_val), verbose=2, callbacks=callbacks, shuffle = True)

    else:
        history = model.fit(x = X_train, y = y_train, epochs=config.EPOCHS, batch_size=batch_size_i,
        validation_data=(X_val, y_val), verbose=2, callbacks=callbacks, shuffle = True)

    # Save test and validation loss in dictionary
    min_index = np.argmin(history.history['val_loss'])
    train_loss = history.history['loss'][min_index]
    val_loss = history.history['val_loss'][min_index]

    model_dict['train_loss'] = train_loss
    model_dict['val_loss'] = val_loss
    model_dict['epochs'] = len(history.history['loss'])

    # Compute pvfps and save accuracy
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)
    pvfp_train_pred = data_postprocessing.calculate_stoch_pvfp(pred_train)
    pvfp_train_target = data_postprocessing.calculate_stoch_pvfp(y_train)
    pvfp_val_pred = data_postprocessing.calculate_stoch_pvfp(pred_val)
    pvfp_val_target = data_postprocessing.calculate_stoch_pvfp(y_val)

    model_dict['train_pvfp_error'] = abs(pvfp_train_target-pvfp_train_pred)
    model_dict['val_pvfp_error'] = abs(pvfp_val_target-pvfp_val_pred)

    number_scenarios = int(pred_train.size / config.PROJECTION_TIME)
    pvfps_train_pred = [data_postprocessing.calculate_pvfp(pred_train, scenario) for scenario in range(number_scenarios)]
    pvfps_train_pred = np.array(pvfps_train_pred)

    pvfps_train_target = [data_postprocessing.calculate_pvfp(y_train, scenario) for scenario in range(number_scenarios)]
    pvfps_train_target = np.array(pvfps_train_target)

    model_dict['train_mae_pvfps'] = np.mean(np.abs(pvfps_train_target - pvfps_train_pred))

    number_scenarios = int(pred_val.size / config.PROJECTION_TIME)
    pvfps_val_pred = [data_postprocessing.calculate_pvfp(pred_val, scenario) for scenario in range(number_scenarios)]
    pvfps_val_pred = np.array(pvfps_val_pred)

    pvfps_val_target = [data_postprocessing.calculate_pvfp(y_val, scenario) for scenario in range(number_scenarios)]
    pvfps_val_target = np.array(pvfps_val_target)

    model_dict['val_mae_pvfps'] = np.mean(np.abs(pvfps_val_target - pvfps_val_pred))

    results.append(model_dict)


    # Save history object
    with open(model_path + '/history.pickle', 'wb') as f:
        pickle.dump(history.history, f)

    # Plot history of losses
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.savefig(model_path + '/loss_epochs.pdf')

# Save history object
with open(model_path + '/model_dict.pickle', 'wb') as f:
    pickle.dump(model_dict, f)

# Write results dictionary into csv file
headers = ['model_name', 'train_loss', 'val_loss','epochs','train_pvfp_error','val_pvfp_error','train_mae_pvfps','val_mae_pvfps']
with open(path + 'results123.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames= headers)
    writer.writeheader()
    writer.writerows(results)
