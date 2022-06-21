# Load predictions from pickle file
import os
from distutils.command.config import config
import pickle
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from sklearn import model_selection
import config
import data_postprocessing

# Settings
plot_val_data = True
plot_test_data = True

# Set plot font
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
            'font.size' : 11,
            'font.family' : 'lmodern'}
plt.rcParams.update(params)

# Load predictions and targets
filepath = config.MODEL_PATH + '/data.pickle'
# model_path = 'grid_search_lstm_gross_surplus/Ensemble_5'
# filepath = model_path + '/data.pickle'

print('Evaluating model ', os.path.dirname(filepath))

if (os.path.exists(filepath)):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
else: 
    data = []
# Load keras history dictionary
filepath = config.MODEL_PATH + '/history.pickle'
# filepath = model_path + '/history.pickle'

if (os.path.exists(filepath)):
    with open(filepath, 'rb') as f:
        history = pickle.load(f)


    print(history.keys())
    index = np.argmin(history['val_loss'])
    if config.USE_ADDITIONAL_INPUT:
        val_mae = 0.5 * history['val_net_profit_head_mae'][index] + 0.5 * history['val_additional_input_head_mae'][index]
        print('val_mse_net_profit: ', history['val_net_profit_head_loss'][index])
        print('val_mae_net_profit: ', history['val_net_profit_head_mae'][index])
        print('train_mse_net_profit: ', history['net_profit_head_loss'][index])
        print('train_mae_net_profit: ', history['net_profit_head_mae'][index])
    else:
        val_mse = history['val_loss'][index]
        val_mae = history['val_mae'][index]
        train_mse = history['loss'][index]
        train_mae = history['mae'][index]

        print('val_mse: ', val_mse)
        print('val_mae: ', val_mae)
        print('train_mse: ', train_mse)
        print('train_mae: ', train_mae)

    print('Epochs: ', len(history['loss']))

pred_test = data[0]
y_test = data[1]

pred_test_original = data[2]
y_test_original = data[3]

if len(data) > 4:
    pred_val = data[4]
    y_val = data[5]
    pred_val_original = data[6]
    y_val_original = data[7]

# Count y_test_original == 0 vs. pred_original == 0
print('# of zeros in targets: ', np.count_nonzero(y_test_original == 0))
print('# of zeros in pred: ', np.count_nonzero(np.absolute(pred_test_original) < 10**-4))

# Find where targets are zero and find out which timestep that corresponds to
indices = np.where(y_test_original == 0)[0]

# Compare to predictions at those indices
pred_test_zero = pred_test_original[indices]
print('pred_zero: ', pred_test_zero[:6])
indices = np.mod(indices, config.PROJECTION_TIME)

# plt.figure(0)
# plt.hist(indices, bins = config.PROJECTION_TIME)
# plt.xlabel('t')
# plt.ylabel('Frequenz')
# plt.title('Histogramm der Nulleinträge (Testdaten)')

# Plot loss per scenario
if plot_test_data:
    x_scen_test = range(int(len(y_test_original) / config.PROJECTION_TIME))

    plt.figure(1)
    plt.plot(x_scen_test, data_postprocessing.calculate_loss_per_scenario(y_test_original, pred_test_original), '.')
    plt.xlabel('Szenario')
    plt.ylabel('MSE')
    plt.title('MSE pro Szenario (Testdaten)')

if plot_val_data:
    x_scen_val = range(int(len(y_val_original) / config.PROJECTION_TIME))
    plt.figure(2)
    plt.plot(x_scen_val, data_postprocessing.calculate_loss_per_scenario(y_val_original, pred_val_original), '.')
    plt.xlabel('Szenario')
    plt.ylabel('MSE')
    plt.title('MSE pro Szenario (Validierungsdaten)')

# Plot MSE for each timestep. Should be higher in first 10 years.
x_time = range(1,config.PROJECTION_TIME+1)

if plot_test_data:
    plt.figure(3)
    plt.plot(x_time, data_postprocessing.calculate_loss_per_timestep(y_test_original, pred_test_original))
    plt.xlabel('t')
    plt.ylabel('MSE')
    plt.title('MSE pro Zeitschritt (Testdaten)')

    plt.figure(4)
    plt.plot(x_time, data_postprocessing.calculate_loss_per_timestep(y_test_original, pred_test_original, loss_metric='mae'))
    plt.xlabel('t')
    plt.ylabel('MAE')
    plt.title('MAE pro Zeitschritt (Testdaten)')
    # plt.savefig(os.path.dirname(filepath) + '/mae_timestep_test.pdf')

if plot_val_data:
    plt.figure(5)
    plt.plot(x_time, data_postprocessing.calculate_loss_per_timestep(y_val_original, pred_val_original))
    plt.xlabel('t')
    plt.ylabel('MSE')
    plt.title('MSE pro Zeitschritt (Validierungsdaten)')
    plt.savefig(os.path.dirname(filepath) + '/mse_timestep_val.pdf')

    plt.figure(6)
    plt.plot(x_time, data_postprocessing.calculate_loss_per_timestep(y_val_original, pred_val_original, loss_metric='mae'))
    plt.xlabel('t')
    plt.ylabel('MAE')
    plt.title('MAE pro Zeitschritt (Validierungsdaten)')
    # plt.savefig(os.path.dirname(filepath) + '/mae_timestep_val.pdf')


# Parity plot
if config.OUTPUT_ACTIVATION == 'tanh' or config.OUTPUT_ACTIVATION == 'linear':
    axis_min, axis_max = -1, 1
elif config.OUTPUT_ACTIVATION == 'sigmoid' or config.OUTPUT_ACTIVATION == 'relu':
    axis_min, axis_max = 0, 1


# plt.figure(7)
# plt.scatter(y_test, pred_test, alpha=0.7)
# plt.plot([axis_min,axis_max], [axis_min,axis_max], 'k--')
# # plt.xlim((axis_min, axis_max))
# # plt.ylim((axis_min, axis_max))
# plt.xlabel('Beobachtung')
# plt.ylabel('Vorhersage')
# # plt.title('Modellvorhersagen vs. Beobachtungen (Testdaten)')
# plt.savefig(os.path.dirname(filepath) + '/parity_test.png')

# Parity plot original scale
if plot_test_data:
    min = np.min(y_test_original)
    max = np.max(y_test_original)
    plt.figure(8)
    plt.scatter(y_test_original, pred_test_original, alpha=0.7)
    plt.plot([min,max], [min,max], 'k--')
    plt.xlabel('Beobachtung')
    plt.ylabel('Vorhersage')
    plt.title('Modellvorhersagen vs. Beobachtungen (Testdaten) (Originalskala)')
    plt.savefig(os.path.dirname(filepath) + '/parity_test_original.pdf')

if plot_val_data:
    min = np.min(y_val_original)
    max = np.max(y_val_original)
    plt.figure(9)
    plt.scatter(y_val_original, pred_val_original, alpha=0.7)
    plt.plot([min,max], [min,max], 'k--')
    # plt.xlim((axis_min, axis_max))
    # plt.ylim((axis_min, axis_max))
    plt.xlabel('Beobachtung')
    plt.ylabel('Zielwert')
    # plt.title('Modellvorhersagen vs. Beobachtungen (Validierungsdaten)')
    plt.savefig(os.path.dirname(filepath) + '/parity_val_original.png')


# Compute residuals
if plot_test_data:
    res = y_test - pred_test
    plt.figure(10)
    plt.hist(res, bins='auto', range=(-0.03,0.03))
    plt.title('Histogram der Residuen (Testdaten)')
    plt.savefig(os.path.dirname(filepath) + '/hist_res_test.pdf')

    res_original = y_test_original - pred_test_original
    plt.figure(11)
    plt.hist(res_original, bins='auto', range=(-100000000,100000000))
    plt.title('Histogram der Residuen (Testdaten) (Originalskala)')
    plt.savefig(os.path.dirname(filepath) + '/hist_res_test_original.pdf')

    # Boxplot of residuals
    plt.figure(12)
    plt.boxplot(res_original)
    plt.title('Boxplot der Residuen (Testdaten)')

if plot_val_data:
    res = y_val - pred_val
    plt.figure(13)
    plt.hist(res, bins='auto', range=(-0.03,0.03))
    plt.title('Histogram der Residuen (Validierungsdaten)')
    # plt.savefig(os.path.dirname(filepath) + '/hist_res_val.pdf')

    res_original = y_val_original - pred_val_original
    plt.figure(14)
    plt.hist(res_original, bins='auto', range=(-100000000,100000000))
    plt.title('Histogram der Residuen (Validierungsdaten) (Originalskala)')
    # plt.savefig(os.path.dirname(filepath) + '/hist_res_val_original.pdf')

    # Boxplot of residuals
    plt.figure(15)
    plt.boxplot(res_original)
    plt.title('Boxplot der Residuen (Validierungsdaten)')

print('Mean of residuals: ', np.mean(res_original))
print('Min: ', np.min(res_original), ' Max: ', np.max(res_original))

print('Range of observations: ', np.min(y_test_original), ' - ', np.max(y_test_original))
print('Range of predictions: ', np.min(pred_test_original), ' - ', np.max(pred_test_original))

# Plot accuracy
x = range(1,config.PROJECTION_TIME+1)

if plot_test_data:
    predictions_np_mean_test = data_postprocessing.calculate_mean_per_timestep(pred_test_original, config.PROJECTION_TIME)
    observations_np_mean_test = data_postprocessing.calculate_mean_per_timestep(y_test_original, config.PROJECTION_TIME)

    plt.figure(16)
    plt.plot(x, predictions_np_mean_test, '.', label = 'Vorhersage')
    plt.plot(x, observations_np_mean_test, 'x', label = 'Zielwert')
    plt.xlabel('t')
    plt.ylabel('Net Profit')
    plt.legend()

if plot_val_data:
    predictions_np_mean_val = data_postprocessing.calculate_mean_per_timestep(pred_val, config.PROJECTION_TIME)
    observations_np_mean_val = data_postprocessing.calculate_mean_per_timestep(y_val, config.PROJECTION_TIME)

    plt.figure(17)
    plt.plot(x, predictions_np_mean_val, '.', label = 'Vorhersage')
    plt.plot(x, observations_np_mean_val, 'x', label = 'Zielwert')
    plt.xlabel('t')
    plt.ylabel('Net Profit')
    plt.legend()
    plt.savefig(os.path.dirname(filepath) + '/accuracy_val.pdf')


# Distribution of predictions vs. targets
if plot_test_data:
    plt.figure(18)
    plt.hist(y_test, bins=1000, alpha = 0.7, label='Zielwert')
    plt.hist(pred_test, bins=1000, alpha= 0.7, label='Vorhersage')
    plt.legend()
    plt.title('Verteilung Zielwerte vs. Vorhersagen (Testdaten)')

    plt.figure(19)
    plt.hist(y_test_original, bins=1000, range=(-0.2*1e8,0.9*1e8), alpha = 0.7, label='Zielwert')
    plt.hist(pred_test_original, bins=1000, range=(-0.2*1e8,0.9*1e8), alpha= 0.7, label='Vorhersage')
    plt.legend()
    plt.title('Verteilung Zielwerte vs. Vorhersagen (Testdaten)(Originalskala)')


# Distribution of predictions vs. targets
if plot_val_data:
    plt.figure(20)
    plt.hist(y_val, bins='auto', range=(0.9,1), alpha = 0.7, label='Beobachtung')
    plt.hist(pred_val, bins='auto', range=(0.9,1), alpha = 0.7, label='Vorhersage')
    # plt.hist(y_val, bins='auto', alpha = 0.7, label='Beobachtung')
    # plt.hist(pred_val, bins='auto', alpha = 0.7, label='Vorhersage')
    plt.legend()
    plt.ylim(top = 10000)
    plt.title('Verteilung Zielwerte vs. Vorhersagen (Validierungsdaten)')
    # plt.savefig(os.path.dirname(filepath) + '/distribution_val.pdf')

    plt.figure(21)
    plt.hist(y_val_original, bins='auto', range=(-1e7,1e8), alpha = 0.7, label='Beobachtung')
    plt.hist(pred_val_original, bins='auto', range=(-1e7,1e8), alpha = 0.7, label='Vorhersage')
    plt.legend()
    plt.ylim(top = 10000)
    plt.title('Verteilung Zielwerte vs. Vorhersagen (Validierungsdaten)(Originalskala)')
    # plt.savefig(os.path.dirname(filepath) + '/distribution_val_original.pdf')

# Plot one specific scenario
scenario_number = 4

pred_val_s = pred_val[scenario_number * config.PROJECTION_TIME : scenario_number * config.PROJECTION_TIME + config.PROJECTION_TIME]
y_val_s = y_val[scenario_number * config.PROJECTION_TIME : scenario_number * config.PROJECTION_TIME + config.PROJECTION_TIME]
plt.figure(22)
plt.plot(x, pred_val_s, '.', label='Vorhersage')
plt.plot(x, y_val_s, 'x', label='Zielwert')
plt.xlabel('t')
plt.ylabel('Net Profit')
plt.legend()
plt.savefig(os.path.dirname(filepath) + '/accuracy_val_scenario.pdf')


from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

print('Some metrics for the Test Data:')
# MSE
mse = mean_squared_error(y_test, pred_test)
print('MSE = ', mse)

mse_orig = mean_squared_error(y_test_original, pred_test_original)
print('MSE (original): ', mse_orig)

# Mean absolute error
mae = mean_absolute_error(y_test, pred_test)
print('MAE = ', mae)

# Compute Mean Absolute Percentage Error (MAPE)
# sk_mape = mean_absolute_percentage_error(y, pred_test) * 100
# print('MAPE: ', sk_mape)
mape = np.mean(np.abs((y_test - pred_test) / y_test)) * 100
print('MAPE = ', mape)

# Symmetric Mean Absolute Percentage Error (SMAPE)
smape = np.mean((np.abs(y_test - pred_test)) / ((np.abs(y_test) + np.abs(pred_test)) / 2)) * 100
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
score = r2_score(y_test, pred_test)
print('R-squared: ', score)

# explained_variance = explained_variance_score(y, pred_test)
# print('Explained variance score: ', explained_variance)

plt.show()
