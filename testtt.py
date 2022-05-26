# TEST
import os
import pickle
from matplotlib import projections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import config
from data_postprocessing import calculate_stochastic_pvfp, calculate_stoch_pvfp, calculate_pvfp, create_discount_vector
import data_preprocessing

# TODO: Analyse mit Trainingsdaten durchfuehren (Testset wahrscheinlich zu klein!)

# Get discount functions from scenario file
input = pd.read_csv(config.PATH_SCENARIO, skiprows=6)

# Remove timestep 60 as the outputs only go to 59
input = input[input['Zeit'] != config.PROJECTION_TIME + 1]
input = input[input['Zeit'] != config.PROJECTION_TIME + 2]
input = input[input['Zeit'] != 0]
discount_functions = input.loc[:,'Diskontfunktion']

# Get discount functions for test and training data
discount_functions = np.array(discount_functions)
test_ratio = (1 - config.TRAIN_RATIO) / 2
idx = (config.TRAIN_RATIO + test_ratio) * (len(discount_functions)/config.PROJECTION_TIME)
idx = int(idx) * config.PROJECTION_TIME
print('idx: ', idx)
discount_functions_train = discount_functions[:idx]
discount_functions = discount_functions[idx:]

# Create discount vector for training and test data
spot_rates = input.loc[:,'1']
spot_rates = np.array(spot_rates)

# convert spot rates to zcb prices: zcb_price[t] = (1+spot_rate[t])^-t
spot_rates_train = spot_rates[:idx]
spot_rates_test = spot_rates[idx:]

discount_vector_9750 = create_discount_vector(spot_rates, 9750)
number_scenarios_test = int(spot_rates_test.size / config.PROJECTION_TIME)
discount_vector_test = [create_discount_vector(spot_rates_test, scenario) for scenario in range(number_scenarios_test)]
discount_vector_test = np.array(discount_vector_test)
print('discount_vector_test size: ', discount_vector_test.size)
# print(discount_vector_9750[:5])

# Load predictions and targets
filepath = config.MODEL_PATH + '/data.pickle'

if (os.path.exists(filepath)):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

# Get test predictions and targets (original scale)
net_profits_pred = data[2]
net_profits_target = data[3]
# Get train predictions and targets (scaled)
net_profits_pred_train = data[4]
net_profits_target_train = data[5]

# Calculate PVFP using discount vector
print('shape net_profit_target: ', net_profits_target.shape)
net_profits_target_9750 = net_profits_target[:config.PROJECTION_TIME]
print('length net_profit_9750: ', len(net_profits_target_9750))
pvfp_9750 = calculate_pvfp(net_profits_target_9750, discount_vector_9750, 0)
print('pvfp_9750: ', pvfp_9750)

print('net_profits_target: ', net_profits_target[:5])

pvfps_test = [calculate_pvfp(net_profits_target, discount_vector_test, scenario) for scenario in range(number_scenarios_test)]
pvfps_test = np.array(pvfps_test)
print(pvfps_test[:5])

plt.figure(0)
plt.hist(pvfps_test, bins='auto')
plt.title('Distribution target PVFPs test data')
plt.show()