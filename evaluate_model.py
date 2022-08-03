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
plot_val_data = False
plot_test_data = True
# metrics_timesteps = ['mae','mape','adj-mae','r-squared']
metrics = ['mae']

# Set plot font
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
            'font.size' : 18,
            'font.family' : 'lmodern',
            'legend.handletextpad': 0.1}
plt.rcParams.update(params)

# Load predictions and targets
model_path = config.MODEL_PATH
# model_path = 'models_60_10k/lstm_basis'
# model_path = 'grid_search_lstm_gross_surplus/Ensemble_5'
filepath = model_path + '/data.pickle'

print('Evaluating model ', os.path.dirname(filepath))

if (os.path.exists(filepath)):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
else: 
    data = []

# Read data from dictionary
pred_train = data["pred_train"]
y_train = data["y_train"]
pred_train_original = data["pred_train_original"]
y_train_original = data["y_train_original"]

pred_val = data["pred_val"]
y_val = data["y_val"]
pred_val_original = data["pred_val_original"]
y_val_original = data["y_val_original"]

pred_test = data["pred_test"]
y_test = data["y_test"]
pred_test_original = data["pred_test_original"]
y_test_original = data["y_test_original"]

# Load keras history dictionary
filepath = model_path + '/history.pickle'

if (os.path.exists(filepath)):
    with open(filepath, 'rb') as f:
        history = pickle.load(f)


    print(history.keys())
    index = np.argmin(history['val_loss'])
    if config.USE_ADDITIONAL_INPUT:
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

    # Plot loss over epochs
    # plt.figure()
    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.ylabel('MSE')
    # plt.xlabel('Epoche')
    # plt.legend(['Training', 'Validierung'], handletextpad = 0.8)
    # plt.tight_layout()
    # plt.savefig(os.path.dirname(filepath) + '/loss_epochs.pdf')

    print('Epochs: ', len(history['loss']))


# Count y_test_original == 0 vs. pred_original == 0
print('# of zeros in targets: ', np.count_nonzero(y_test_original <= 10e-4))
print('# of zeros in pred: ', np.count_nonzero(np.absolute(pred_test_original) < 10e-4))

# Find where targets are zero and find out which timestep that corresponds to
indices = np.where(y_test_original == 0)[0]

# Compare to predictions at those indices
pred_test_zero = pred_test_original[indices]
print('pred_zero: ', pred_test_zero[:6])
indices = np.mod(indices, config.PROJECTION_TIME)

plt.figure()
plt.hist(indices, bins = config.PROJECTION_TIME)
plt.xlabel('t')
plt.ylabel('Frequenz')
plt.title('Histogramm der Nulleinträge (Testdaten)')

# Plot MSE for each timestep. Should be higher in first 10 years.
x_time = range(1,config.PROJECTION_TIME+1)

if plot_test_data:

    x_scen_test = range(int(len(y_test_original) / config.PROJECTION_TIME))


    for metric in metrics:
        plt.figure()
        plt.plot(x_scen_test, data_postprocessing.calculate_loss_per_scenario(y_test, pred_test, loss_metric=metric), '.')
        plt.xlabel('Szenario')
        plt.ylabel(metric)
        plt.tight_layout()

    for metric in metrics:
        if metric == 'mape': # MAPE can only be calculated on scaled data
            loss, stds = data_postprocessing.calculate_loss_per_timestep(y_test, pred_test, loss_metric=metric)
        else:
            loss, stds = data_postprocessing.calculate_loss_per_timestep(y_test_original, pred_test_original, loss_metric=metric)

        # plt.figure()
        fig, ax = plt.subplots()
        plt.plot(x_time, loss)
        plt.xlabel('$t$')
        plt.xticks([0,10,20,30,40,50,60])
        if metric == 'r-squared':
            plt.ylabel('$R^2$', rotation=0, labelpad=15)
            plt.ylim(0,1)
            ax_ins = ax.inset_axes([0.1, 0.1, 0.3, 0.3])
            ax_ins.plot(x_time, loss)
            ax_ins.set_xticklabels([])
            ax_ins.set_yticks([-3,0])
        else:
            # plt.ylabel(metric.upper() + ' (\%)')
            plt.ylabel(metric.upper())
        # ax_ins = fig.add_axes([0.5, 0.5, 0.47, 0.47])
        plt.tight_layout()
        # plt.savefig(model_path + f'/{metric}_timesteps_test.pdf')

        print(f'Max of {metric}: \t{np.max(loss)}')
        print(f'... at index {np.argmax(loss)}')
        print(f'Min of {metric}: \t{np.min(loss)}')
        index_min = np.argmin(loss)
        print(f'... at index {index_min}')
        print(f'... where the std is: {stds[index_min]}')

        zero_count_y, zero_count_pred = data_postprocessing.count_zeros_per_timestep(y_test_original, pred_test_original)
        print(f'Zero count of targets at {index_min}: \t{zero_count_y[index_min]}')
        print(f'Zero count of predictions at {index_min}: \t{zero_count_pred[index_min]}')


        plt.figure()
        # plt.hist(stds[0], bins='auto')
        plt.plot(x_time, stds)
        plt.xlabel('$t$')
        plt.ylabel('$\sigma$')
        plt.tight_layout()
        plt.savefig(model_path + '/std_timesteps_test.pdf')

        # Plot correlation of MAE and sigma
        plt.figure()
        plt.scatter(stds, loss, alpha=0.5)
        plt.xlabel("$\sigma$")
        plt.ylabel("MAE")
        # plt.xscale("log")
        plt.tight_layout()

        # print("Correlation between MAE and sigma: ", np.corrcoef(loss, stds, rowvar=False))
        

if plot_val_data:

    x_scen_val = range(int(len(y_val_original) / config.PROJECTION_TIME))

    # for metric in metrics:
    #     plt.figure()
    #     # plt.plot(x_time, data_postprocessing.calculate_loss_per_timestep(y_test_original, pred_test_original, loss_metric=metric))
    #     plt.plot(x_scen_val, data_postprocessing.calculate_loss_per_scenario(y_val, pred_val, loss_metric=metric), '.')
    #     plt.xlabel('Szenario')
    #     plt.ylabel(metric)
    #     plt.tight_layout()

    for metric in metrics:
        if metric == 'mape': # MAPE can only be calculated on scaled data
            loss, stds = data_postprocessing.calculate_loss_per_timestep(y_val+1, pred_val+1, loss_metric=metric)

            # loss, stds = data_postprocessing.calculate_loss_per_timestep(y_val_original, pred_val_original, loss_metric=metric)
        else:
            loss, stds = data_postprocessing.calculate_loss_per_timestep(y_val_original, pred_val_original, loss_metric=metric)

        # plt.figure()
        # plt.plot(x_time, loss)
        # plt.xlabel('t')
        # plt.ylabel(metric.upper())
        # plt.tight_layout()
        # plt.savefig(model_path + f'/{metric}_timesteps.pdf')

        fig, ax = plt.subplots()
        plt.plot(x_time, loss)
        plt.xlabel('$t$')
        plt.xticks([0,10,20,30,40,50,60])
        if metric == 'r-squared':
            plt.ylabel('$R^2$', rotation=0, labelpad=15)
            plt.ylim(0,1)
        else:
            # plt.ylabel(metric.upper() + ' (\%)')
            plt.ylabel(metric.upper())
        plt.tight_layout()
        # ax_ins = fig.add_axes([0.5, 0.5, 0.47, 0.47])
        # ax_ins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        # ax_ins.plot(x_time, loss)
        # ax_ins.set_xticklabels([])
        # ax_ins.set_yticks([0,10,20,30])
        plt.savefig(model_path + f'/{metric}_timesteps_val.pdf')

        print(f'Max of {metric}: \t{np.max(loss)}')
        print(f'... at index {np.argmax(loss)}')
        print(f'Min of {metric}: \t{np.min(loss)}')
        print(f'... at index {np.argmin(loss)}')
        print(f'... where the std is: {stds[np.argmin(loss)]}')


        plt.figure()
        # plt.hist(stds[0], bins='auto')
        plt.plot(x_time, stds)
        plt.xlabel('t')
        plt.ylabel('$\sigma$')
        plt.tight_layout()
        plt.savefig(model_path + '/std_timesteps.pdf')


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
    plt.xlabel('Zielwert')
    plt.ylabel('Vorhersage')
    # plt.title('Modellvorhersagen vs. Beobachtungen (Testdaten) (Originalskala)')
    plt.tight_layout()
    plt.savefig(os.path.dirname(filepath) + '/parity_test_original.jpeg', dpi=800)

if plot_val_data:
    min = np.min(y_val_original)
    max = np.max(y_val_original)
    plt.figure(9)
    plt.scatter(y_val_original, pred_val_original, alpha=0.7)
    plt.plot([min,max], [min,max], 'k--')
    # plt.xlim((axis_min, axis_max))
    # plt.ylim((axis_min, axis_max))
    plt.xlabel('Zielwert')
    plt.ylabel('Vorhersage')
    # plt.title('Modellvorhersagen vs. Beobachtungen (Validierungsdaten)')
    plt.tight_layout()
    plt.savefig(os.path.dirname(filepath) + '/parity_val_original.jpeg', dpi = 800)


# Compute residuals
if plot_test_data:
    res = y_test - pred_test
    plt.figure(10)
    plt.hist(res, bins='auto', range=(-0.03,0.03))
    plt.title('Histogram der Residuen (Testdaten)')
    plt.tight_layout()
    plt.savefig(os.path.dirname(filepath) + '/hist_res_test.pdf')

    res_original = y_test_original - pred_test_original
    plt.figure(11)
    plt.hist(res_original, bins='auto', range=(-100000000,100000000))
    # plt.title('Histogram der Residuen (Testdaten) (Originalskala)')
    plt.xlabel('Residuum')
    plt.ylabel('Frequenz')
    plt.tight_layout()
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
    plt.tight_layout()
    # plt.savefig(os.path.dirname(filepath) + '/hist_res_val.pdf')

    res_original = y_val_original - pred_val_original
    plt.figure(14)
    plt.hist(res_original, bins='auto', range=(-100000000,100000000))
    plt.title('Histogram der Residuen (Validierungsdaten) (Originalskala)')
    plt.tight_layout()
    # plt.savefig(os.path.dirname(filepath) + '/hist_res_val_original.pdf')

    # Boxplot of residuals
    plt.figure(15)
    plt.boxplot(res_original)
    plt.title('Boxplot der Residuen (Validierungsdaten)')

# print('Mean of residuals: ', np.mean(res_original))
# print('Min: ', np.min(res_original), ' Max: ', np.max(res_original))

# print('Range of observations: ', np.min(y_test_original), ' - ', np.max(y_test_original))
# print('Range of predictions: ', np.min(pred_test_original), ' - ', np.max(pred_test_original))

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
    plt.tight_layout()
    plt.savefig(model_path + '/accuracy_test.pdf')
    plt.legend()

if plot_val_data:
    predictions_np_mean_val = data_postprocessing.calculate_mean_per_timestep(pred_val_original, config.PROJECTION_TIME)
    observations_np_mean_val = data_postprocessing.calculate_mean_per_timestep(y_val_original, config.PROJECTION_TIME)

    plt.figure(17)
    plt.plot(x, predictions_np_mean_val, '.', label = 'Vorhersage')
    plt.plot(x, observations_np_mean_val, 'x', label = 'Zielwert')
    plt.xlabel('t')
    plt.ylabel('Net Profit')
    plt.legend()
    plt.tight_layout()
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
    plt.tight_layout()
    # plt.savefig(os.path.dirname(filepath) + '/distribution_val.pdf')

    plt.figure(21)
    plt.hist(y_val_original, bins='auto', range=(-1e7,1e8), alpha = 0.7, label='Beobachtung')
    plt.hist(pred_val_original, bins='auto', range=(-1e7,1e8), alpha = 0.7, label='Vorhersage')
    plt.legend()
    plt.ylim(top = 10000)
    plt.title('Verteilung Zielwerte vs. Vorhersagen (Validierungsdaten)(Originalskala)')
    plt.tight_layout()
    # plt.savefig(os.path.dirname(filepath) + '/distribution_val_original.pdf')

# Plot one specific scenario
if plot_val_data:
    scenario_number = 4

    pred_val_s = pred_val_original[scenario_number * config.PROJECTION_TIME : scenario_number * config.PROJECTION_TIME + config.PROJECTION_TIME]
    y_val_s = y_val_original[scenario_number * config.PROJECTION_TIME : scenario_number * config.PROJECTION_TIME + config.PROJECTION_TIME]
    plt.figure(22)
    plt.plot(x, pred_val_s, '.', label='Vorhersage')
    plt.plot(x, y_val_s, 'x', label='Zielwert')
    plt.xlabel('t')
    plt.ylabel('Net Profit')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.dirname(filepath) + '/accuracy_val_scenario.pdf')

# Compute Mean, Max, Min, Std etc
pred_test_mean = np.mean(pred_test_original)
pred_test_median = np.median(pred_test_original)
pred_test_min = np.min(pred_test_original)
pred_test_max = np.max(pred_test_original)
pred_test_std = np.std(pred_test_original)
pred_test_sem = data_postprocessing.calculate_sem(pred_test_original)

print('Exploratory Analysis Predicted Net Profits:')
print("Min: \t\t", pred_test_min / 10e6, " Mio.")
print("Max: \t\t", pred_test_max / 10e6, " Mio.")
print("Mean: \t\t", pred_test_mean / 10e6, " Mio.")
print("Median: \t\t", pred_test_median / 10e6, " Mio.")
print("Std: \t\t", pred_test_std / 10e6, " Mio.")
print("Standard error of the mean: \t", pred_test_sem)

# Compute Mean, Max, Min, Std etc
y_test_mean = np.mean(y_test_original)
y_test_median = np.median(y_test_original)
y_test_min = np.min(y_test_original)
y_test_max = np.max(y_test_original)
y_test_std = np.std(y_test_original)
y_test_sem = data_postprocessing.calculate_sem(y_test_original)


print('Exploratory Analysis Target Net Profits:')
print("Min: \t\t", y_test_min / 10e6, " Mio.")
print("Max: \t\t", y_test_max / 10e6, " Mio.")
print("Mean: \t\t", y_test_mean / 10e6, " Mio.")
print("Median: \t\t", y_test_median / 10e6, " Mio.")
print("Std: \t\t", y_test_std / 10e6, " Mio.")
print("Standard error of the mean: \t", y_test_sem)

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

print('--------- TEST DATA ----------')
# MSE
mse = mean_squared_error(y_test, pred_test)
print('MSE_test = ', mse)

# Mean absolute error
mae = mean_absolute_error(y_test, pred_test)
print('MAE_test = ', mae)
mae_orig = mean_absolute_error(y_test_original, pred_test_original)
print('MAE_test (original) = ', mae_orig)

mape = np.mean(np.abs((y_test - pred_test) / y_test)) * 100
print('MAPE_test = ', mape)

# Symmetric Mean Absolute Percentage Error (SMAPE)
smape = np.mean((np.abs(y_test - pred_test)) / ((np.abs(y_test) + np.abs(pred_test)) / 2)) * 100
print('SMAPE_test = ', smape)

smape_orig = np.mean((np.abs(y_test_original - pred_test_original)) / ((np.abs(y_test_original) + np.abs(pred_test_original)) / 2)) * 100
print('SMAPE_test (original) = ', smape_orig)

# R-squared
score = r2_score(y_test, pred_test)
print('R-squared: ', score)

# explained_variance = explained_variance_score(y, pred_test)
# print('Explained variance score: ', explained_variance)

# Metrics on validation data
if len(y_val) > 0:
    print('--------- VALIDATION DATA ----------')
    # MSE
    mse = mean_squared_error(y_val, pred_val)
    print('MSE_val = ', mse)

    # RMSE
    rmse = mean_squared_error(y_val, pred_val, squared=False)
    print('RMSE_val = ', rmse)
    rmse_orig = mean_squared_error(y_val_original, pred_val_original, squared=False)
    print('RMSE_val (original) = ', rmse_orig)

    # Mean absolute error
    mae = mean_absolute_error(y_val, pred_val)
    print('MAE_val = ', mae)
    mae_orig = mean_absolute_error(y_val_original, pred_val_original)
    print('MAE_val (original) = ', mae_orig)

    mape = np.mean(np.abs((y_val - pred_val) / y_val)) * 100
    print('MAPE_val = ', mape)

if len(y_train) > 0:
    print('--------- TRAINING DATA ----------')
    # MSE
    mse = mean_squared_error(y_train, pred_train)
    print('MSE_train = ', mse)

    # RMSE
    rmse = mean_squared_error(y_train, pred_train, squared=False)
    print('RMSE_train = ', rmse)
    rmse_orig = mean_squared_error(y_train_original, pred_train_original, squared=False)
    print('RMSE_train (original) = ', rmse_orig)

    # Mean absolute error
    mae = mean_absolute_error(y_train, pred_train)
    print('MAE_train = ', mae)
    mae_orig = mean_absolute_error(y_train_original, pred_train_original)
    print('MAE_train (original) = ', mae_orig)

    mape = np.mean(np.abs((y_train - pred_train) / y_train)) * 100
    print('MAPE_train = ', mape)

plt.show()
