# TEST
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import config

features = np.array([
    [[0,0,0,0],[0,0,0,0],[0,0,0,0]],
    [[1,1,1,1],[1,1,1,1],[1,1,1,1]],
    [[2,2,2,2],[2,2,2,2],[2,2,2,2]],
    [[3,3,3,3],[3,3,3,3],[3,3,3,3]],
    [[4,4,4,4],[4,4,4,4],[4,4,4,4]]
    ])
labels = np.array([0,1,2,3,4])

print(features.shape)
print(labels.shape)

print('BEFORE SHUFFLE:')
print(features)
print(labels)

features, labels = shuffle(features, labels, random_state=config.RANDOM_SEED)

print('AFTER SHUFFLE:')
print(features)
print(labels)

x = np.tile([1,2,3,4,5],3)
y = np.array(range(0,15))

print('x:')
print(x)
print(y)

window_x = np.concatenate([x[0:2], y[0:2]])
print(window_x)