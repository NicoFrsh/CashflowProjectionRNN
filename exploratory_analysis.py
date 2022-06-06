# Exploratory Analysis
import pandas as pd
import numpy as np
import config
import matplotlib.pyplot as plt

input = pd.read_csv(config.PATH_SCENARIO, skiprows=6)
output = pd.read_csv(config.PATH_OUTPUT)

