# Data inspection
# Evaluate saved model
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt

import config
import data_preprocessing
import model
import data_postprocessing

# Read inputs

X_train, y_train, X_test, y_test, scaler_output = data_preprocessing.prepare_data(
    config.PATH_SCENARIO, config.PATH_OUTPUT, config.OUTPUT_VARIABLE, shuffle_data=False)

y_test_mean = data_postprocessing.calculate_mean(y_test, 59)

