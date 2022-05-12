# Load predictions from pickle file
import os
from distutils.command.config import config
import pickle
from turtle import position
from typing import IO
import numpy as np
import matplotlib.pyplot as plt
import config
import data_postprocessing


# This code is to load predictions and targets 
if config.USE_ADDITIONAL_INPUT:
    filepath = 'models/bs_{0}_{1}_{2}_{3}_{4}/data.pickle'.format(config.BATCH_SIZE, config.MODEL_TYPE, str.replace(config.ADDITIONAL_INPUT, ' ', '_'), config.LSTM_LAYERS, config.LSTM_CELLS)
else:
    filepath = 'models/test_bs_{0}_{1}_{2}_{3}/data.pickle'.format(config.BATCH_SIZE, config.MODEL_TYPE, config.LSTM_LAYERS, config.LSTM_CELLS)

with open(filepath, 'rb') as f:
    data = pickle.load(f)

if config.USE_ADDITIONAL_INPUT:
    filepath = 'models/bs_{0}_{1}_{2}_{3}_{4}/history.pickle'.format(config.BATCH_SIZE, config.MODEL_TYPE, str.replace(config.ADDITIONAL_INPUT, ' ', '_'), config.LSTM_LAYERS, config.LSTM_CELLS)
else:
    filepath = 'models/test_bs_{0}_{1}_{2}_{3}/history.pickle'.format(config.BATCH_SIZE, config.MODEL_TYPE, config.LSTM_LAYERS, config.LSTM_CELLS)

with open(filepath, 'rb') as f:
    history = pickle.load(f)

print('Evaluating model ', os.path.dirname(filepath))

print(history.keys())
print('val_mse: ', np.min(history['val_loss']))
if config.USE_ADDITIONAL_INPUT:
    index = np.argmin(history['val_loss'])
    val_mae = 0.5 * history['val_net_profit_head_mae'][index] + 0.5 * history['val_additional_input_head_mae'][index]
    # val_mae =  index WHERE val_loss = min --> 0.5 * val_net_profit_head_mae[index] + 0.5 val_additional_input_head_mae[index] 
    print('val_mse_net_profit: ', history['val_net_profit_head_loss'][index])
    print('val_mae_net_profit: ', history['val_net_profit_head_mae'][index])
else:
    val_mae = np.min(history['val_mae'])

print('val_mae: ', val_mae)

pred = data[0]
y = data[1]
pred_original = data[2]
y_original = data[3]
if len(data) > 4:
    pred_train = data[4]
    y_train = data[5]

# TODO: Count y_original == 0 vs. pred_original == 0
print('# of zeros in targets: ', np.count_nonzero(y_original == 0))
print('# of zeros in pred: ', np.count_nonzero(pred_original==0))

# Find where targets are zero and find out which timestep that corresponds to
indices = np.where(y_original == 0)[0]

# Compare to predictions at those indices
pred_zero = pred_original[indices]
print('pred_zero: ', pred_zero[:6])
# modulo = [59 for i in indices]
# modulo = np.array(modulo)
indices = np.mod(indices, 59)

print('indices: ', indices[:10])

# Plot MSE for each timestep. Should be higher in first 10 years.
mse_over_timesteps = data_postprocessing.calculate_loss_per_timestep(y, pred)
mae_over_timesteps = data_postprocessing.calculate_loss_per_timestep(y, pred, loss_metric='mae')
x = range(1,60)

plt.figure(21)
plt.plot(x, mse_over_timesteps)
plt.title('MSE over timesteps')

plt.figure(22)
plt.plot(x, mae_over_timesteps)
plt.title('MAE over timesteps')

plt.figure(20)
plt.hist(indices)
plt.title('Histogram of zeros-entries')

# Parity plot
plt.figure(0)
plt.scatter(y, pred, alpha=0.7)
plt.plot([-1,1], [-1,1], 'k--')
plt.xlabel('Observations')
plt.ylabel('Predictions')
plt.title('Parity Plot Test Data')
plt.savefig(os.path.dirname(filepath) + '/parity.png')

# Parity plot original scale
min = np.min(y_original)
max = np.max(y_original)
plt.figure(9)
plt.scatter(y_original, pred_original, alpha=0.7)
plt.plot([min,max], [min,max], 'k--')
plt.xlabel('Observations')
plt.ylabel('Predictions')
plt.title('Parity Plot Test Data Original Scale')

# Compute residuals
# TODO: Use original scale to obtain better interpretability
res = y - pred
plt.figure(1)
plt.hist(res, bins='auto', range=(-0.03,0.03))
plt.title('Histogram of residuals')

res_original = y_original - pred_original
plt.figure(2)
plt.hist(res_original, bins=100, range=(-100000000,100000000))
plt.title('Histogram of residuals (original scale)')

print('Mean of residuals: ', np.mean(res_original))
print('Min: ', np.min(res_original), ' Max: ', np.max(res_original))

print('Range of observations: ', np.min(y_original), ' - ', np.max(y_original))
print('Range of predictions: ', np.min(pred_original), ' - ', np.max(pred_original))

predictions_np_mean = data_postprocessing.calculate_mean_per_timestep(pred_original, 59)
observations_np_mean = data_postprocessing.calculate_mean_per_timestep(y_original, 59)

x = range(1,60)
plt.figure(3)
plt.plot(x, predictions_np_mean, '.', label = 'Predictions')
plt.plot(x, observations_np_mean, 'x', label = 'Observations')
plt.xlabel('year')
plt.ylabel(config.OUTPUT_VARIABLE)
plt.title('Average of predictions vs. observations')
plt.legend()
plt.savefig(os.path.dirname(filepath) + '/accuracy.png')


# Boxplot of residuals
plt.figure(4)
plt.boxplot(res_original)
plt.title('Boxplot of residuals')

# Distribution of predictions vs. targets
plt.figure(5)
plt.hist(y_original, bins=1000, range=(-20000000,100000000))
plt.hist(pred_original, bins=1000, range=(-20000000,100000000))
plt.legend(loc = 'upper left')
plt.title('Distribution y_target vs. y_pred')

# Distribution of predictions vs. targets
if len(data) > 4:
    plt.figure(6)
    plt.hist(y_train, bins='auto', range=(0.8,1))
    plt.hist(pred_train, bins='auto', range=(0.8,1))
    plt.legend(loc = 'upper left')
    plt.title('Distribution y_train vs. pred_train')

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# MSE
mse = mean_squared_error(y, pred)
print('MSE = ', mse)

mse_orig = mean_squared_error(y_original, pred_original)
print('MSE (original): ', mse_orig)

# Mean absolute error
mae = mean_absolute_error(y, pred)
print('MAE = ', mae)

# Compute Mean Absolute Percentage Error (MAPE)
sk_mape = mean_absolute_percentage_error(y, pred) * 100
print('MAPE: ', sk_mape)
mape = np.mean(np.abs((y - pred) / y)) * 100
print('MAPE = ', mape)

# Symmetric Mean Absolute Percentage Error (SMAPE)
smape = np.mean((np.abs(y - pred)) / ((np.abs(y) + np.abs(pred)) / 2)) * 100
print('SMAPE = ', smape)

smape_2 = np.mean((np.abs(y - pred)) / (np.abs(y) + np.abs(pred))) * 100
print('adj. SMAPE = ', smape_2)

# Mean Absolute Scaled Error (MASE)

# Mean Directional Accuracy (MDA)

# Logarithm of acurracy ratio
# log_acc = np.mean(np.log(y / pred))
# print('log acc. = ', log_acc)

# MRAE 
y_shifted = np.roll(y,1)
y_shifted[0] = 0
mrae = np.mean(np.abs((y - pred) / (y - y_shifted))) * 100
print('MRAE = ', mrae)

# R-squared
score = r2_score(y, pred)
print('R-squared: ', score)

explained_variance = explained_variance_score(y, pred)
print('Explained variance score: ', explained_variance)

plt.show()
