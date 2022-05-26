# Hyperparameter search using keras tuner
import numpy as np
from tensorflow import keras
import keras_tuner

hp = keras_tuner.HyperParameters()
model_type = hp.Choice('model_type', ['simple_rnn', 'lstm', 'gru'])
learning_rate = hp.Float('learning_rate', ['0.01','0.005', '0.001'])

