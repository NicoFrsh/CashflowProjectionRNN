import numpy as np
from tensorflow import keras
import pickle
import pandas as pd

grid_search_path = './grid_search_lstm_gross_surplus'

results = pd.read_excel(grid_search_path+'/results_lstm_gru.xlsx')

number_models = 5

for i in range(number_models):

    model = 0