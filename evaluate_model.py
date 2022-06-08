# Load predictions from pickle file
import os
from distutils.command.config import config
import pickle
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import config
import data_postprocessing

# Set plot font
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
            'font.size' : 11,
            'font.family' : 'lmodern'}
plt.rcParams.update(params)

# Load predictions and targets
filepath = config.MODEL_PATH + '/data.pickle'

if (os.path.exists(filepath)):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
else: 
    data = []
# Load keras history dictionary
filepath = config.MODEL_PATH + '/history.pickle'

with open(filepath, 'rb') as f:
    history = pickle.load(f)

print('Evaluating model ', os.path.dirname(filepath))

print(history.keys())
print('val_mse: ', np.min(history['val_loss']))
if config.USE_ADDITIONAL_INPUT:
    index = np.argmin(history['val_loss'])
    val_mae = 0.5 * history['val_net_profit_head_mae'][index] + 0.5 * history['val_additional_input_head_mae'][index]
    print('val_mse_net_profit: ', history['val_net_profit_head_loss'][index])
    print('val_mae_net_profit: ', history['val_net_profit_head_mae'][index])
else:
    val_mae = np.min(history['val_mae'])

print('val_mae: ', val_mae)

pred_test = data[0]
y = data[1]

pred_test_original = data[2]
y_test_original = data[3]
if len(data) > 4:
    pred_train = data[4]
    y_train = data[5]

# Count y_test_original == 0 vs. pred_original == 0
print('# of zeros in targets: ', np.count_nonzero(y_test_original == 0))
print('# of zeros in pred: ', np.count_nonzero(np.absolute(pred_test_original) < 10**-4))

# Find where targets are zero and find out which timestep that corresponds to
indices = np.where(y_test_original == 0)[0]

# Compare to predictions at those indices
pred_test_zero = pred_test_original[indices]
print('pred_zero: ', pred_test_zero[:6])
indices = np.mod(indices, config.PROJECTION_TIME)

plt.figure(20)
plt.hist(indices, bins = config.PROJECTION_TIME)
plt.xlabel('t')
plt.ylabel('Frequenz')
plt.title('Histogramm der Nulleinträge (Testdaten)')

# Plot MSE for each timestep. Should be higher in first 10 years.
mse_over_timesteps = data_postprocessing.calculate_loss_per_timestep(y, pred_test)
mae_over_timesteps = data_postprocessing.calculate_loss_per_timestep(y, pred_test, loss_metric='mae')
x_time = range(1,config.PROJECTION_TIME+1)
x_scen = range(int(len(y) / config.PROJECTION_TIME))

print('length y_test: ', len(y))

plt.figure(19)
plt.plot(x_scen, data_postprocessing.calculate_loss_per_scenario(y, pred_test), '.')
plt.xlabel('Szenario')
plt.ylabel('MSE')
plt.title('MSE pro Szenario (Testdaten)')

plt.figure(21)
plt.plot(x_time, mse_over_timesteps)
plt.xlabel('t')
plt.ylabel('MSE')
plt.title('MSE pro Zeitschritt (Testdaten)')

plt.figure(22)
plt.plot(x_time, mae_over_timesteps)
plt.xlabel('t')
plt.ylabel('MAE')
plt.title('MAE pro Zeitschritt (Testdaten)')

plt.figure(23)
plt.plot(x_time, data_postprocessing.calculate_loss_per_timestep(y_train, pred_train, loss_metric='mse'))
plt.xlabel('t')
plt.ylabel('MSE')
# plt.title('MSE pro Zeitschritt (Trainingsdaten)')
plt.savefig(os.path.dirname(filepath) + '/mse_timestep_train.pdf')

# Parity plot
if config.OUTPUT_ACTIVATION == 'tanh' or config.OUTPUT_ACTIVATION == 'linear':
    axis_min, axis_max = -1, 1
elif config.OUTPUT_ACTIVATION == 'sigmoid' or config.OUTPUT_ACTIVATION == 'relu':
    axis_min, axis_max = 0, 1

plt.figure(0)
plt.scatter(y, pred_test, alpha=0.7)
plt.plot([axis_min,axis_max], [axis_min,axis_max], 'k--')
# plt.xlim((axis_min, axis_max))
# plt.ylim((axis_min, axis_max))
plt.xlabel('Beobachtung')
plt.ylabel('Vorhersage')
# plt.title('Modellvorhersagen vs. Beobachtungen (Testdaten)')
plt.savefig(os.path.dirname(filepath) + '/parity.pdf')

plt.figure(999)
plt.scatter(y_train, pred_train, alpha=0.7)
plt.plot([axis_min,axis_max], [axis_min,axis_max], 'k--')
# plt.xlim((axis_min, axis_max))
# plt.ylim((axis_min, axis_max))
plt.xlabel('Beobachtung')
plt.ylabel('Vorhersage')
# plt.title('Modellvorhersagen vs. Beobachtungen (Trainingsdaten)')
plt.savefig(os.path.dirname(filepath) + '/parity_train.png')

# Parity plot original scale
min = np.min(y_test_original)
max = np.max(y_test_original)
plt.figure(9)
plt.scatter(y_test_original, pred_test_original, alpha=0.7)
plt.plot([min,max], [min,max], 'k--')
plt.xlabel('Beobachtung')
plt.ylabel('Vorhersage')
# plt.title('Modellvorhersagen vs. Beobachtungen (Testdaten) (Originalskala)')
plt.savefig(os.path.dirname(filepath) + '/parity_original.pdf')


# Compute residuals
res = y - pred_test
plt.figure(1)
plt.hist(res, bins='auto', range=(-0.03,0.03))
plt.title('Histogram der Residuen (Testdaten)')

res_original = y_test_original - pred_test_original
plt.figure(2)
plt.hist(res_original, bins='auto', range=(-100000000,100000000))
plt.title('Histogram der Residuen (Testdaten) (Originalskala)')
plt.savefig(os.path.dirname(filepath) + '/hist_res_original.pdf')

print('Mean of residuals: ', np.mean(res_original))
print('Min: ', np.min(res_original), ' Max: ', np.max(res_original))

print('Range of observations: ', np.min(y_test_original), ' - ', np.max(y_test_original))
print('Range of predictions: ', np.min(pred_test_original), ' - ', np.max(pred_test_original))

# predictions_np_mean_test = data_postprocessing.calculate_mean_per_timestep(pred_test_original, config.PROJECTION_TIME)
# observations_np_mean_test = data_postprocessing.calculate_mean_per_timestep(y_test_original, config.PROJECTION_TIME)
predictions_np_mean_train = data_postprocessing.calculate_mean_per_timestep(pred_train, config.PROJECTION_TIME)
observations_np_mean_train = data_postprocessing.calculate_mean_per_timestep(y_train, config.PROJECTION_TIME)

x = range(1,config.PROJECTION_TIME+1)

# plt.figure(998)
# plt.plot(x, predictions_np_mean_test, '.', label = 'Vorhersage')
# plt.plot(x, observations_np_mean_test, 'x', label = 'Beobachtung')
# plt.xlabel('t')
# plt.ylabel('Net Profit')
# plt.legend()

plt.figure(3)
plt.plot(x, predictions_np_mean_train, '.', label = 'Vorhersage')
plt.plot(x, observations_np_mean_train, 'x', label = 'Beobachtung')
plt.xlabel('t')
plt.ylabel('Net Profit')
plt.legend()
plt.savefig(os.path.dirname(filepath) + '/accuracy.pdf')


# Boxplot of residuals
plt.figure(4)
plt.boxplot(res_original)
plt.title('Boxplot der Residuen')

# Distribution of predictions vs. targets
plt.figure(5)
plt.hist(y_test_original, bins=1000, range=(-0.2*1e8,0.9*1e8), alpha = 0.7, label='Beobachtung')
plt.hist(pred_test_original, bins=1000, range=(-0.2*1e8,0.9*1e8), alpha= 0.7, label='Vorhersage')
plt.legend()
plt.title('Verteilung Beobachtung vs. Vorhersage (Testdaten)(Originalskala)')

# Distribution of predictions vs. targets
if len(data) > 4:
    plt.figure(6)
    plt.hist(y_train, bins='auto', range=(0.9,1), alpha = 0.7, label='Beobachtung')
    plt.hist(pred_train, bins='auto', range=(0.9,1), alpha = 0.7, label='Vorhersage')
    # plt.hist(y_train, bins='auto', alpha = 0.7, label='Beobachtung')
    # plt.hist(pred_train, bins='auto', alpha = 0.7, label='Vorhersage')
    plt.legend()
    plt.ylim(top = 10000)
    plt.title('Verteilung Beobachtung vs. Vorhersage (Trainingsdaten)')
    plt.savefig(os.path.dirname(filepath) + '/distribution.pdf')

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

print('Some metrics for the Test Data:')
# MSE
mse = mean_squared_error(y, pred_test)
print('MSE = ', mse)

mse_orig = mean_squared_error(y_test_original, pred_test_original)
print('MSE (original): ', mse_orig)

# Mean absolute error
mae = mean_absolute_error(y, pred_test)
print('MAE = ', mae)

# Compute Mean Absolute Percentage Error (MAPE)
# sk_mape = mean_absolute_percentage_error(y, pred_test) * 100
# print('MAPE: ', sk_mape)
mape = np.mean(np.abs((y - pred_test) / y)) * 100
print('MAPE = ', mape)

# Symmetric Mean Absolute Percentage Error (SMAPE)
smape = np.mean((np.abs(y - pred_test)) / ((np.abs(y) + np.abs(pred_test)) / 2)) * 100
print('SMAPE = ', smape)

# smape_2 = np.mean((np.abs(y - pred_test)) / (np.abs(y) + np.abs(pred_test))) * 100
# print('adj. SMAPE = ', smape_2)

# Mean Absolute Scaled Error (MASE)

# Mean Directional Accuracy (MDA)

# Logarithm of acurracy ratio
# log_acc = np.mean(np.log(y / pred))
# print('log acc. = ', log_acc)

# MRAE 
# y_shifted = np.roll(y,1)
# y_shifted[0] = 0
# mrae = np.mean(np.abs((y - pred_test) / (y - y_shifted))) * 100
# print('MRAE = ', mrae)

# R-squared
score = r2_score(y, pred_test)
print('R-squared: ', score)

# explained_variance = explained_variance_score(y, pred_test)
# print('Explained variance score: ', explained_variance)

plt.show()
