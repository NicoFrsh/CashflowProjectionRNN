# Load predictions from pickle file
from distutils.command.config import config
import pickle
import numpy as np
import matplotlib.pyplot as plt
import config

filepath = './models/model_acc_{0}_{1}/data.pickle'.format(config.LSTM_LAYERS, config.LSTM_CELLS)

with open(filepath, 'rb') as f:
    data = pickle.load(f)

pred = data[0]
y = data[1]
print(type(pred))
print(len(pred))
print(pred[:10])
print(type(y))
print(len(y))
print(y[:10])

# Parity plot
plt.figure(0)
plt.scatter(y, pred, alpha=0.7)
plt.plot([-1,1], [-1,1], 'k--')
plt.xlabel('Observations')
plt.ylabel('Predictions')
plt.title('Parity Plot Test Data')
plt.show()