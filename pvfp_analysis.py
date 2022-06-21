# TEST
import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
from data_postprocessing import calculate_stochastic_pvfp, calculate_stoch_pvfp, calculate_pvfp
import data_preprocessing

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
# filepath = config.MODEL_PATH + '/data.pickle'
filepath = 'grid_search_lstm_gross_surplus/Ensemble_5/data.pickle'

if (os.path.exists(filepath)):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

# Get test predictions and targets (original scale)
net_profits_pred = data[2]
print('len(net_profits_test): ', len(net_profits_pred))
net_profits_target = data[3]
# Get train predictions and targets (original scale)
net_profits_pred_train = data[6]
net_profits_target_train = data[7]

print('net_profits shape: ', net_profits_pred_train.shape)
print('net_profits_target shape: ', net_profits_target_train.shape)

# Calculate PVFP
# 1. version
pvfp_pred = calculate_stoch_pvfp(net_profits_pred, discount_functions_test)
pvfp_target = calculate_stoch_pvfp(net_profits_target, discount_functions_test)

print('------ Test Data ------')
print('PVFP prediction: ', pvfp_pred)
print('PVFP target: ', pvfp_target)
print('Difference: ', abs(pvfp_target - pvfp_pred))

# 2. version
# pvfp_pred = calculate_stoch_pvfp(net_profits_pred, discount_functions_test)
# pvfp_target = calculate_stoch_pvfp(net_profits_target, discount_functions_test)

# print('PVFP prediction: ', pvfp_pred)
# print('PVFP target: ', pvfp_target)
# print('Difference: ', abs(pvfp_target - pvfp_pred))

# Calculate PVFP (Train Data)
pvfp_pred_train = calculate_stoch_pvfp(net_profits_pred_train, discount_functions_train)
pvfp_target_train = calculate_stoch_pvfp(net_profits_target_train, discount_functions_train)

print('------ Train Data ------')
print('PVFP prediction: ', pvfp_pred_train)
print('PVFP target: ', pvfp_target_train)
print('Difference: ', abs(pvfp_target_train - pvfp_pred_train)) 

# 2. version
# pvfp_pred_train = calculate_stoch_pvfp(net_profits_pred_train, discount_functions_train)
# pvfp_target_train = calculate_stoch_pvfp(net_profits_target_train, discount_functions_train)

# print('PVFP prediction: ', pvfp_pred_train)
# print('PVFP target: ', pvfp_target_train)
# print('Difference: ', abs(pvfp_target_train - pvfp_pred_train))

# Compute PVFP for each scenario and compare distributions (wie Akho) using test set
number_scenarios = int(net_profits_pred.size / config.PROJECTION_TIME)
pvfps_pred = [calculate_pvfp(net_profits_pred, scenario, discount_functions_test) for scenario in range(number_scenarios)]
pvfps_pred = np.array(pvfps_pred)

pvfps_target = [calculate_pvfp(net_profits_target, scenario, discount_functions_test) for scenario in range(number_scenarios)]
pvfps_target = np.array(pvfps_target)
print('len(pvfps_target): ', len(pvfps_target))
print('pvfps_target.head: ', pvfps_target[:5])

# Calculate mean absolute error of test pvfps
mae = np.mean(np.abs(pvfps_target - pvfps_pred))
print('MAE (PVFPs): ', mae)

# Plot distribution
plt.figure(0)
plt.hist(pvfps_target, bins=30, alpha = 0.5, label='Targets')
plt.hist(pvfps_pred, bins=30, alpha= 0.5, label='Predictions')
plt.legend()
plt.title('Distribution target PVFPs vs. predicted PVFPs (Test Data)')

# Compute PVFP for each scenario and compare distributions (wie Akho) using training set
number_scenarios_train = int(net_profits_pred_train.size / config.PROJECTION_TIME)
pvfps_pred_train = [calculate_pvfp(net_profits_pred_train, scenario, discount_functions_train) for scenario in range(number_scenarios_train)]
pvfps_pred_train = np.array(pvfps_pred_train)

pvfps_target_train = [calculate_pvfp(net_profits_target_train, scenario, discount_functions_train) for scenario in range(number_scenarios_train)]
pvfps_target_train = np.array(pvfps_target_train)

# Plot distribution
plt.figure(1)
plt.hist(pvfps_target_train, bins='auto', alpha = 0.5, label='Targets')
plt.hist(pvfps_pred_train, bins='auto', alpha= 0.5, label='Predictions')
plt.legend()
plt.title('Distribution target PVFPs vs. predicted PVFPs (Training Data)')
plt.show()
