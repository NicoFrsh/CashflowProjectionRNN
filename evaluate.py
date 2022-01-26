# Evaluate saved model
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import config
import data_preprocessing
import model
import data_postprocessing

# Parameters
# TODO: Global steuern, ob net profit oder additional input geplottet werden soll. Oder unnötig? (Immer nur net profit relevant)
if config.USE_ADDITIONAL_INPUT:
    model_path = 'models/model_{}_1_32.h5'.format(str.replace(config.ADDITIONAL_INPUT, ' ', '_'))
else:
    model_path = 'models/model_yearly_tanh_1_32.h5'
plot_test_accuracy = True
plot_train_accuracy = True
plot_test_mse = True
plot_train_mse = True
plot_test_mae = True
plot_train_mae = True
plot_test_mse_per_scenario = True
plot_train_mse_per_scenario = True
plot_test_mae_per_scenario = False
plot_train_mae_per_scenario = False


# Create training and test data
if config.USE_ADDITIONAL_INPUT:
    X_train, y_train, y_2_train, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)
else:
    X_train, y_train, X_test, y_test, scaler_output, scaler_input = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)

# Input shape = (timesteps, # features)
lstm_input_shape = (config.TIMESTEPS, X_train.shape[2])

# Get average of labels (used as initial bias value)
average_label = np.mean(y_train)
average_label_2 = None
if config.USE_ADDITIONAL_INPUT:
    average_label_2 = np.mean(y_2_train)

model_lstm = model.create_rnn_model(lstm_input_shape, average_label, average_label_2)

model_lstm.summary()

# Load trained model
model_lstm.load_weights(model_path)

# # Evaluate network - Get test and training accuracy
if config.USE_ADDITIONAL_INPUT:
    score_test = model_lstm.evaluate(X_test, [y_test, y_2_test], verbose=0)
    score_train = model_lstm.evaluate(X_train, [y_train, y_2_train], verbose=0)
else:
    score_test = model_lstm.evaluate(X_test, y_test, verbose=0)
    score_train = model_lstm.evaluate(X_train, y_train, verbose=0)

print(f'Test loss: {score_test[0]} / Test mae: {score_test[1]}')
print(f'Training loss: {score_train[0]} / Training mae: {score_train[1]}')

# Make predictions
# If an additional input, e.g. RfB, is used the net shall predict the test data using the predicted additional inputs recursively.
if config.USE_ADDITIONAL_INPUT:
    predictions_np, predictions_additional_input = data_postprocessing.recursive_prediction(X_test, model_lstm)
    # predictions_np, predictions_additional_input = model_lstm.predict(X_test)
    predictions_np_train, predictions_additional_input_train = model_lstm.predict(X_train)
else:
    # predictions_np = data_postprocessing.recursive_prediction(X_test, model_lstm)
    predictions_np = model_lstm.predict(X_test)
    predictions_np_train = model_lstm.predict(X_train)

# Inverse scaling of outputs
y_test_original = scaler_output.inverse_transform(y_test)
predictions_np_inverted = scaler_output.inverse_transform(predictions_np)

predictions_np_mean = data_postprocessing.calculate_mean_per_timestep(predictions_np_inverted, 59)
observations_np_mean = data_postprocessing.calculate_mean_per_timestep(y_test_original, 59)

if config.USE_ADDITIONAL_INPUT:
    y_2_test_original = scaler_additional_input.inverse_transform(y_2_test)
    predictions_additional_input_inverted = scaler_additional_input.inverse_transform(predictions_additional_input)

    predictions_additional_input_mean = data_postprocessing.calculate_mean_per_timestep(predictions_additional_input_inverted, 59)
    observations_additional_input_mean = data_postprocessing.calculate_mean_per_timestep(y_2_test_original, 59)

# print('Range y_test:')
# print('Min: ', min(y_test_original), ' Max: ', max(y_test_original))
# print('Range predictions:')
# print('Min: ', min(predictions_inverted), ' Max: ', max(predictions_inverted))


if plot_test_accuracy:
    x = range(1,60)
    plt.figure(0)
    plt.plot(x, predictions_np_mean, '.', label = 'Predictions')
    plt.plot(x, observations_np_mean, 'x', label = 'Observations')
    plt.xlabel('year')
    plt.ylabel(config.OUTPUT_VARIABLE)
    plt.title('Average of predictions vs. observations')
    plt.legend()



if plot_train_accuracy:

    training_predictions = predictions_np_train

    # Revert scaling
    training_predictions_inverted = scaler_output.inverse_transform(training_predictions)
    training_observations_original = scaler_output.inverse_transform(y_train)

    training_predictions_mean = data_postprocessing.calculate_mean_per_timestep(training_predictions_inverted, 59)
    training_observations_mean = data_postprocessing.calculate_mean_per_timestep(training_observations_original, 59)

    x = range(1,60)

    plt.figure(1)
    plt.plot(x, training_predictions_mean, '.', label = 'Predictions')
    plt.plot(x, training_observations_mean, 'x', label = 'Observations')
    plt.xlabel('year')
    plt.ylabel(config.OUTPUT_VARIABLE)
    plt.title('Training: Average of predictions vs. observations')
    plt.legend()

    # Plot one specific scenario
    scenario_number = 2

    predictions_scenario = training_predictions_inverted[scenario_number * 59 : scenario_number * 59 + 59]
    observations_scenario = training_observations_original[scenario_number * 59 : scenario_number * 59 + 59]

    plt.figure(2)
    plt.plot(x, predictions_scenario, '.', label = 'Predictions')
    plt.plot(x, observations_scenario, 'x', label = 'Observations')
    plt.xlabel('year')
    plt.ylabel(config.OUTPUT_VARIABLE)
    plt.title('Training: Scenario {} Predictions vs. observations'.format(scenario_number))
    plt.legend()

# Calculate test loss
if plot_test_mse:
    test_loss = data_postprocessing.calculate_loss_per_timestep(y_test, predictions_np)
    x = range(1,60)

    plt.figure(3)
    plt.plot(x, test_loss)
    plt.xlabel('year')
    plt.ylabel('MSE')
    plt.title('Test MSE over time')

if plot_train_mse:    
    train_loss = data_postprocessing.calculate_loss_per_timestep(y_train, predictions_np_train)
    x = range(1,60)

    plt.figure(4)
    plt.plot(x, train_loss)
    plt.xlabel('year')
    plt.ylabel('MSE')
    plt.title('Train MSE over time')

if plot_test_mae:
    test_loss = data_postprocessing.calculate_loss_per_timestep(y_test, predictions_np, loss_metric='mae')
    x = range(1,60)

    plt.figure(5)
    plt.plot(x, test_loss)
    plt.xlabel('year')
    plt.ylabel('MAE')
    plt.title('Test MAE over time')

if plot_train_mae:
    train_loss = data_postprocessing.calculate_loss_per_timestep(y_train, predictions_np_train, loss_metric='mae')
    x = range(1,60)

    plt.figure(6)
    plt.plot(x, train_loss)
    plt.xlabel('year')
    plt.ylabel('MAE')
    plt.title('Train MAE over time')

if plot_test_mse_per_scenario:
    print('Test MSE')
    test_loss = data_postprocessing.calculate_loss_per_scenario(y_test, predictions_np)
    x = range(len(test_loss))

    plt.figure(7)
    plt.boxplot(test_loss)
    # plt.xlabel('Scenario')
    plt.ylabel('MSE')
    plt.title('Boxplot: Test MSE over scenarios')

if plot_train_mse_per_scenario:
    print('Train MSE')
    train_loss = data_postprocessing.calculate_loss_per_scenario(y_train, predictions_np_train)
    x = range(len(train_loss))

    plt.figure(8)
    plt.boxplot(train_loss)
    # plt.xlabel('Scenario')
    plt.ylabel('MSE')
    plt.title('Boxplot: Train MSE over scenarios')

if plot_test_mae_per_scenario:
    print('Test MAE')
    test_loss = data_postprocessing.calculate_loss_per_scenario(y_test, predictions_np, loss_metric='mae')
    x = range(len(test_loss))

    plt.figure(9)
    plt.plot(x, test_loss, '.')
    plt.xlabel('Scenario')
    plt.ylabel('MAE')
    plt.title('Test MAE over scenarios')

if plot_train_mae_per_scenario:
    print('Train MAE')
    train_loss = data_postprocessing.calculate_loss_per_scenario(y_train, predictions_np_train, loss_metric='mae')
    x = range(len(train_loss))

    plt.figure(10)
    plt.plot(x, train_loss, '.')
    plt.xlabel('Scenario')
    plt.ylabel('MAE')
    plt.title('Train MAE over scenarios')

plt.show()