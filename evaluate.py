# Evaluate saved model
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt

import config
import data_preprocessing
import model
import data_postprocessing

# Parameters
plot_test_accuracy = True
plot_train_accuracy = True

# Read inputs
X_train, y_train, X_test, y_test, scaler_output = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)

# Input shape = (timesteps, # features)
lstm_input_shape = (config.TIMESTEPS, X_train.shape[2])

model_lstm = model.create_rnn_model(lstm_input_shape)

model_lstm.load_weights('models/mymodel.h5')

# # Evaluate network
score = model_lstm.evaluate(X_test, y_test, verbose=0)
print(f'Test loss: {score[0]} / Test mae: {score[1]}')

# Make predictions
predictions = model_lstm.predict(X_test)

print('Range y_test:')
print('Min: ', min(y_test), ' Max: ', max(y_test))
print('Range predictions:')
print('Min: ', min(predictions), ' Max: ', max(predictions))

# Inverse scaling of outputs
y_test_original = scaler_output.inverse_transform(y_test)
predictions_inverted = scaler_output.inverse_transform(predictions)

print('Range y_test:')
print('Min: ', min(y_test_original), ' Max: ', max(y_test_original))
print('Range predictions:')
print('Min: ', min(predictions_inverted), ' Max: ', max(predictions_inverted))

predictions_mean = data_postprocessing.calculate_mean(predictions_inverted, 60)
observations_mean = data_postprocessing.calculate_mean(y_test_original, 60)

if plot_test_accuracy:
    plt.figure(0)
    plt.plot(predictions_mean, '.', label = 'Predictions')
    plt.plot(observations_mean, 'x', label = 'Observations')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Predicted vs. observed {}'.format(config.OUTPUT_VARIABLE))
    plt.legend()



if plot_train_accuracy:

    # Get training accuracy
    score = model_lstm.evaluate(X_train, y_train, verbose=0)
    print(f'Training loss: {score[0]} / Training mae: {score[1]}')

    training_predictions = model_lstm.predict(X_train)

    # Revert scaling
    training_predictions_inverted = scaler_output.inverse_transform(training_predictions)
    training_observations_original = scaler_output.inverse_transform(y_train)

    training_predictions_mean = data_postprocessing.calculate_mean(training_predictions_inverted, 60)
    training_observations_mean = data_postprocessing.calculate_mean(training_observations_original, 60)

    plt.figure(1)
    plt.plot(training_predictions_mean, '.', label = 'Predictions')
    plt.plot(training_observations_mean, 'x', label = 'Observations')
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.title('Training: Predicted vs. observed {}'.format(config.OUTPUT_VARIABLE))
    plt.legend()

plt.show()