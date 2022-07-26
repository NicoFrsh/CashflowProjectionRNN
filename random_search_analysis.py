# Analyse Random Search
import time
import os
import numpy as np
import pickle
import pandas as pd
import config
import matplotlib.pyplot as plt

# Set plot font
plt.rcParams['text.latex.preamble']=[r"\usepackage{lmodern}"]
params = {'text.usetex' : True,
            'font.size' : 25,
            'font.family' : 'lmodern'}
plt.rcParams.update(params)

def dot_boxplot(data, **kwargs):
    # fig = plt.figure()
    for i in range(len(data)):
        x = np.random.normal(i+1, 0.04, len(data[i]))
        plt.plot(x, data[i], "b.", alpha=0.3)
    plt.boxplot(data, **kwargs)
    plt.grid()
    plt.tight_layout()

grid_search_path = './grid_search_lstm_gross_surplus/'

results = pd.read_excel(grid_search_path+'results_lstm_gru.xlsx')

mse_lstm, mse_gru = [],[]
mse_T_5, mse_T_10, mse_T_15, mse_T_20 = [],[],[],[]
mse_BS_250, mse_BS_500, mse_BS_1000 = [],[],[]
mse_d_32, mse_d_64, mse_d_128 = [],[],[]
mse_y_activation_tanh, mse_y_activation_linear = [],[]
mse_z_activation_tanh, mse_z_activation_linear = [],[]

for i in range(len(results)):

    model_type = results['model_type'][i]

    # Extract parameters from model name
    parameters = results['model_name'][i].split('_')    
    timesteps = int(parameters[2])
    batch_size = int(parameters[4])
    recurrent_activation = parameters[5]
    y_activation = parameters[6]
    z_activation = parameters[7]
    rnn_layers = int(parameters[8])
    rnn_cells = int(parameters[9])

    val_mse = results['val_mse'][i]

    # MODEL TYPE
    if model_type == 'lstm':
        mse_lstm.append(val_mse)
    elif model_type == 'gru':
        mse_gru.append(val_mse)

    # TIMESTEPS T
    if timesteps == 5:
        mse_T_5.append(val_mse)
    elif timesteps == 10:
        mse_T_10.append(val_mse)
    elif timesteps == 15:
        mse_T_15.append(val_mse)
    elif timesteps == 20:
        mse_T_20.append(val_mse)

    # BATCH SIZE
    if batch_size == 250:
        mse_BS_250.append(val_mse)
    elif batch_size == 500:
        mse_BS_500.append(val_mse)
    elif batch_size == 1000:
        mse_BS_1000.append(val_mse)

    # d
    if rnn_cells == 32:
        mse_d_32.append(val_mse)
    elif rnn_cells == 64:
        mse_d_64.append(val_mse)
    elif rnn_cells == 128:
        mse_d_128.append(val_mse)

    # y-ACTIVATION
    if y_activation == 'tanh':
        mse_y_activation_tanh.append(val_mse)
    elif y_activation == 'linear':
        mse_y_activation_linear.append(val_mse)

    # z-ACTIVATION
    if z_activation == 'tanh':
        mse_z_activation_tanh.append(val_mse)
    elif z_activation == 'linear':
        mse_z_activation_linear.append(val_mse)

# # Boxplot of validation errors
# fig, axs = plt.subplots(2,3, figsize=(10,6), sharex=False, sharey=True, constrained_layout=True)

# # Model Types
# axs[0,0].boxplot([mse_lstm,mse_gru])
# axs[0,0].set_xticks(list(range(1,3)))
# axs[0,0].set_xticklabels(["LSTM","GRU"])
# axs[0,0].set_xlabel("Modelltyp")
# axs[0,0].set_ylabel("Validierungsfehler")
# axs[0,0].grid()

# # Timesteps T
# axs[0,1].boxplot([mse_T_5, mse_T_10, mse_T_15, mse_T_20])
# axs[0,1].set_xticks(list(range(1,5)))
# axs[0,1].set_xticklabels(["5","10","15","20"])
# axs[0,1].set_xlabel("T")
# axs[0,1].grid()

# axs[0,2].boxplot([mse_BS_250, mse_BS_500, mse_BS_1000])
# axs[0,2].set_xticks(list(range(1,4)))
# axs[0,2].set_xticklabels(["250","500","1000"])
# axs[0,2].set_xlabel("Batchgröße")
# axs[0,2].grid()

# axs[1,0].boxplot([mse_d_32, mse_d_64, mse_d_128])
# axs[1,0].set_xticks(list(range(1,4)))
# axs[1,0].set_xticklabels(["32","64","128"])
# axs[1,0].set_xlabel("$d$")
# axs[1,0].set_ylabel("Validierungsfehler")
# axs[1,0].grid()

# axs[1,1].boxplot([mse_y_activation_tanh, mse_y_activation_linear])
# axs[1,1].set_xticks(list(range(1,3)))
# axs[1,1].set_xticklabels(["tanh","linear"])
# axs[1,1].set_xlabel("$\phi_{\hat{y}}$")
# axs[1,1].grid()

# axs[1,2].boxplot([mse_z_activation_tanh, mse_z_activation_linear])
# axs[1,2].set_xticks(list(range(1,3)))
# axs[1,2].set_xticklabels(["tanh","linear"])
# axs[1,2].set_xlabel("$\phi_{\hat{z}}$")
# axs[1,2].grid()

# plt.tight_layout()

# Plot single plots separately
# Model Types
plt.figure()
dot_boxplot([mse_lstm,mse_gru], showmeans=True)
plt.xticks(list(range(1,3)), ["LSTM","GRU"])
plt.xlabel("Modelltyp")
plt.ylabel("MSE")
plt.savefig(grid_search_path + "boxplots_model_type.pdf")

# Timesteps T
plt.figure()
dot_boxplot([mse_T_5,mse_T_10,mse_T_15,mse_T_20], showmeans=True)
plt.xticks(list(range(1,5)), ["5","10","15","20"])
plt.xlabel("$T$")
plt.savefig(grid_search_path + "boxplots_T.pdf")

plt.figure()
dot_boxplot([mse_BS_250, mse_BS_500, mse_BS_1000], showmeans=True)
plt.xticks(list(range(1,4)), ["250","500","1000"])
plt.xlabel("Batchgröße")
plt.savefig(grid_search_path + "boxplots_batch_size.pdf")

plt.figure()
dot_boxplot([mse_d_32, mse_d_64, mse_d_128], showmeans=True)
plt.xticks(list(range(1,4)), ["32","64","128"])
plt.xlabel("$n$")
plt.ylabel("MSE")
plt.savefig(grid_search_path + "boxplots_n.pdf")

plt.figure()
dot_boxplot([mse_y_activation_tanh, mse_y_activation_linear], showmeans=True)
plt.xticks(list(range(1,3)), ["tanh","linear"])
plt.xlabel("$\phi_{\hat{y}}$", labelpad=-0.2)
plt.savefig(grid_search_path + "boxplots_y_activation.pdf")

plt.figure()
dot_boxplot([mse_z_activation_tanh, mse_z_activation_linear], showmeans=True)
plt.xticks(list(range(1,3)), ["tanh","linear"])
plt.xlabel("$\phi_{\hat{z}}$")
plt.savefig(grid_search_path + "boxplots_z_activation.pdf")

plt.show()

