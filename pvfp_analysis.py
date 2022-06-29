# TEST
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
from data_postprocessing import calculate_stochastic_pvfp, calculate_stoch_pvfp, calculate_pvfp
import data_preprocessing

# Set plot font
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
            'font.size' : 18,
            'font.family' : 'lmodern'}
plt.rcParams.update(params)

# Get discount functions from scenario file
input = pd.read_csv(config.PATH_SCENARIO, skiprows=6)

# Remove timestep 60 as the outputs only go to 59
input = input[input['Zeit'] != config.PROJECTION_TIME + 1]
input = input[input['Zeit'] != 0]

# Adjust discount functions
discount_vector = data_preprocessing.generate_discount_vector(input)
input['Diskontfunktion'] = discount_vector

discount_functions = input.loc[:,'Diskontfunktion']


# Get discount functions for test and training data
# TODO: ACHTUNG: Funktioniert nicht, wenn geshuffelt wurde!!
discount_functions = np.array(discount_functions)
test_ratio = (1 - config.TRAIN_RATIO) / 2
idx_test_start = (config.TRAIN_RATIO + test_ratio) * (len(discount_functions)/config.PROJECTION_TIME)
idx_test_start = int(idx_test_start) * config.PROJECTION_TIME
idx_train_end = config.TRAIN_RATIO * (len(discount_functions)/config.PROJECTION_TIME)
idx_train_end = int(idx_train_end) * config.PROJECTION_TIME
discount_functions_train = discount_functions[:idx_train_end]
discount_functions_test = discount_functions[idx_test_start:]

# Load predictions and targets
filepath = config.MODEL_PATH + '/data.pickle'
# filepath = 'grid_search_lstm_gross_surplus/Ensemble_5/data.pickle'

if (os.path.exists(filepath)):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

# Get test predictions and targets (original scale)
# net_profits_pred = data[2]
pred_test = data["pred_test_original"]
# net_profits_target = data[3]
y_test = data["y_test_original"]
# Get train predictions and targets (original scale)
pred_train = data["pred_train_original"]
y_train = data["y_train_original"]


# Calculate PVFP
# 1. version
pvfp_pred_test = calculate_stoch_pvfp(pred_test, discount_functions_test)
pvfp_y_test = calculate_stoch_pvfp(y_test, discount_functions_test)

print('------ Test Data ------')
print('PVFP prediction:\t', pvfp_pred_test)
print('PVFP target:\t\t', pvfp_y_test)
difference = pvfp_y_test - pvfp_pred_test
print('Absolute difference:\t', abs(difference))
print('Relative difference:\t', (difference / abs(pvfp_y_test)) * 100, '%')


# Calculate PVFP (Train Data)
pvfp_pred_train = calculate_stoch_pvfp(pred_train, discount_functions_train)
pvfp_y_train = calculate_stoch_pvfp(y_train, discount_functions_train)

print('------ Train Data ------')
print('PVFP prediction:\t', pvfp_pred_train)
print('PVFP target:\t\t', pvfp_y_train)
difference = pvfp_y_train - pvfp_pred_train
print('Absolut Difference:\t', abs(difference)) 
print('Relative difference:\t', (difference / abs(pvfp_y_train)) * 100, '%')

# Total PVFP (Train + Validation + Test)
pred_val = data["pred_val_original"]
y_val = data["y_val_original"]

pvfp_pred_val = calculate_stoch_pvfp(pred_val)
pvfp_y_val = calculate_stoch_pvfp(y_val)

# pred_total = np.concatenate((pred_train,pred_val,pred_test))
pred_total = np.concatenate((y_train,y_val,pred_test))
y_total = np.concatenate((y_train,y_val,y_test))
pvfp_pred_total = calculate_stoch_pvfp(pred_total)
# pvfp_y_total = (1/3) * (pvfp_y_train + pvfp_y_test + pvfp_y_val)
pvfp_y_total = calculate_stoch_pvfp(y_total)

print('-------- TOTAL PVFP ---------')
print('PVFP prediction:\t', pvfp_pred_total)
print('PVFP target:\t\t', pvfp_y_total)
difference = pvfp_y_total - pvfp_pred_total
print('Absolut Difference:\t', abs(difference)) 
print('Relative difference:\t', (difference / pvfp_y_total) * 100, '%')

print('-------- PVFP (TRAIN + VALIDATION) ---------')
pred_train_val = np.concatenate((pred_train,pred_val))
y_train_val = np.concatenate((y_train,y_val))
pvfp_pred_train_val = calculate_stoch_pvfp(pred_train_val)
pvfp_y_train_val = calculate_stoch_pvfp(y_train_val)
print('PVFP prediction:\t', pvfp_pred_train_val)
print('PVFP target:\t\t', pvfp_y_train_val)
difference = pvfp_y_train_val - pvfp_pred_train_val
print('Absolut Difference:\t', abs(difference)) 
print('Relative difference:\t', (difference / pvfp_y_train_val) * 100, '%')

# Compute PVFP for each scenario and compare distributions (wie Akho) using test set
number_scenarios_test = int(pred_test.size / config.PROJECTION_TIME)
pvfps_pred_test = [calculate_pvfp(pred_test, scenario, discount_functions_test) for scenario in range(number_scenarios_test)]
pvfps_pred_test = np.array(pvfps_pred_test)

pvfps_y_test = [calculate_pvfp(y_test, scenario, discount_functions_test) for scenario in range(number_scenarios_test)]
pvfps_y_test = np.array(pvfps_y_test)
print('len(pvfps_test): ', len(pvfps_y_test))
print('pvfps_target.head: ', pvfps_y_test[:5])

# Calculate mean absolute error of test pvfps
mae = np.mean(np.abs(pvfps_y_test - pvfps_pred_test))
print('MAE (PVFPs): ', mae)

# Plot distribution
plt.figure(0)
plt.hist(pvfps_y_test, bins='auto', alpha = 0.5, label='Zielwerte')
plt.hist(pvfps_pred_test, bins='auto', alpha= 0.5, label='Vorhersagen')
plt.legend()
plt.title('Distribution target PVFPs vs. predicted PVFPs (Test Data)')

# Compute PVFP for each scenario and compare distributions (wie Akho) using training set
number_scenarios_train = int(pred_train.size / config.PROJECTION_TIME)
pvfps_pred_train = [calculate_pvfp(pred_train, scenario, discount_functions_train) for scenario in range(number_scenarios_train)]
pvfps_pred_train = np.array(pvfps_pred_train)

pvfps_y_train = [calculate_pvfp(y_train, scenario, discount_functions_train) for scenario in range(number_scenarios_train)]
pvfps_y_train = np.array(pvfps_y_train)

# Plot distribution
plt.figure(1)
plt.hist(pvfps_y_train, bins='auto', alpha = 0.5, label='Zielwerte')
plt.hist(pvfps_pred_train, bins='auto', alpha= 0.5, label='Vorhersagen')
plt.legend()
plt.title('Distribution target PVFPs vs. predicted PVFPs (Training Data)')

# Plot PVFPs comparison (Test Data)
x_test = np.arange(number_scenarios_test)
plt.figure(2)
# plt.bar(x, pvfps_pred_test)
plt.plot(x_test, pvfps_pred_test, '.', alpha = 0.7, label = 'Vorhersage')
plt.plot(x_test, pvfps_y_test, 'x', alpha = 0.7, label = 'Zielwert')
plt.legend()
plt.title('PVFPs im Vergleich (Test Data)')

# Plot PVFPs comparison (Train Data)
x_train = np.arange(number_scenarios_train)
plt.figure(3)
# plt.bar(x, pvfps_pred_test)
plt.plot(x_train, pvfps_pred_train, '.', alpha = 0.7, label = 'Vorhersage')
plt.plot(x_train, pvfps_y_train, 'x', alpha = 0.7, label = 'Zielwert')
plt.legend()
plt.title('PVFPs im Vergleich (Train Data)')

# Plot SORTED PVFPs comparison (Test Data)
plt.figure(4)
# plt.bar(x, pvfps_pred_test)
plt.plot(x_test, np.sort(pvfps_pred_test), '.', alpha = 0.7, label = 'Vorhersage')
plt.plot(x_test, np.sort(pvfps_y_test), '.', alpha = 0.7, label = 'Zielwert')
plt.legend()
plt.title('PVFPs im Vergleich (Test Data)')

# Plot SORTED PVFPs comparison (Train Data)
plt.figure(5)
# plt.bar(x, pvfps_pred_test)
plt.plot(x_train, np.sort(pvfps_pred_train), '.', alpha = 0.7, label = 'Vorhersage')
plt.plot(x_train, np.sort(pvfps_y_train), '.', alpha = 0.7, label = 'Zielwert')
plt.legend()
plt.title('PVFPs im Vergleich (Train Data)')

# Scatterplots
plt.figure(6)
plt.scatter(pvfps_y_test, pvfps_pred_test)
plt.plot([np.min(pvfps_y_test), np.max(pvfps_y_test)], [np.min(pvfps_y_test), np.max(pvfps_y_test)], 'k--')
plt.xlabel('Zielwerte')
plt.ylabel('Vorhersagen')
plt.tight_layout()
plt.savefig(os.path.dirname(filepath) + "/parity_pvfps_test.pdf")

plt.figure(7)
plt.scatter(pvfps_y_train, pvfps_pred_train)
plt.plot([np.min(pvfps_y_train), np.max(pvfps_y_train)], [np.min(pvfps_y_train), np.max(pvfps_y_train)], 'k--')
plt.xlabel('Zielwerte')
plt.ylabel('Vorhersagen')

# Boxplot of residuals
res_test = pvfps_y_test - pvfps_pred_test
x = np.random.normal(1, 0.005, len(res_test))

# plt.figure(8)
# # plt.plot(x, res_test, ".", alpha = 0.3)
# plt.boxplot(res_test, showfliers=True)
# plt.grid()
# plt.title('Boxplot der Residuen (Testdaten)')
# plt.xticks([])
# plt.tight_layout()

# Histogramm der Residuen
plt.figure(9)
plt.hist(res_test, bins='auto')
# plt.title("Histogramm der Residuen Testdaten")
plt.xlabel("Residuum")
plt.ylabel("Frequenz")
plt.tight_layout()
plt.savefig(os.path.dirname(filepath) + "/hist_res_pvfps_test.pdf")



plt.show()