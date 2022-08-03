# Write predictions and labels to csv
import numpy as np
import pandas as pd
import pickle
import config

# model_path = "grid_search_lstm_gross_surplus/Ensemble_5"
model_path = config.MODEL_PATH

with open(model_path + "/data.pickle" , 'rb') as f:
    data = pickle.load(f)

pred_train = data["pred_train_original"]
y_train = data["y_train_original"]

pred_val = data["pred_val_original"]
y_val = data["y_val_original"]

pred_test = data["pred_test_original"]
y_test = data["y_test_original"]

# Concatenate train and validation data
pred_train = np.concatenate((pred_train, pred_val))
y_train = np.concatenate((y_train, y_val))

# Reshape arrays
number_train_scenarios = int(config.NUMBER_SCENARIOS * config.TRAIN_RATIO)
number_train_scenarios += int(number_train_scenarios * config.VAL_RATIO)
print("num_train_scen: ", number_train_scenarios)
pred_train = pred_train.reshape((number_train_scenarios,config.PROJECTION_TIME))
y_train = y_train.reshape((number_train_scenarios,config.PROJECTION_TIME))

number_test_scenarios = config.NUMBER_SCENARIOS - number_train_scenarios
print("num_test_scen: ", number_test_scenarios)
pred_test = pred_test.reshape((number_test_scenarios,config.PROJECTION_TIME))
y_test = y_test.reshape((number_test_scenarios,config.PROJECTION_TIME))

pred_all = np.concatenate((pred_train, pred_test))
y_all = np.concatenate((y_train, y_test))


# Write to csv
pred_all_df = pd.DataFrame(pred_all)
y_all_df = pd.DataFrame(y_all)

pred_all_df.to_csv(model_path + "/predictions_rnn.csv")
y_all_df.to_csv(model_path + "/labels_aephix.csv")