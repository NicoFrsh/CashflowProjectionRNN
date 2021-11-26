# Data preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import config

def prepare_data(scenario_path, outputs_path, output_variable, recurrent_timesteps = config.TIMESTEPS, shuffle_data = False,
train_ratio = config.TRAIN_RATIO):
    """
    Prepares data and gets it ready to serve as input to LSTM-Neural-Network.
    Parameters:
        - scenario_path: file path for the scenario file containing all parameters that were used to create the output
        - outputs_path: output created by the CFP-Tool
        - output_variable: name of the output that we want to predict with our model
        - recurrent_timesteps: how many previous timesteps are included in the prediction for the current output
    """

    input = pd.read_csv(scenario_path, skiprows=6)
    output = pd.read_csv(outputs_path)

    # Preprocess data
    rfb = output[output['Variable'] == 'RfB']
    output = output[output['Variable'] == output_variable]
    output = output.iloc[:, 0:62]
    rfb = rfb.iloc[:, 0:62]
    # Remove 'Variable' column
    output = output.drop(columns=['Variable'])
    rfb = rfb.drop(columns=['Variable'])
    # Shift Simulation by -1 to match data from input dataframe.
    output['Simulation'] -= 1
    rfb['Simulation'] -= 1
    output = output.rename(columns={'Simulation':'Pfad'})
    rfb = rfb.rename(columns={'Simulation':'Pfad'})

    # Change format of dataframe
    output = pd.melt(output, id_vars=['Pfad'], var_name='Zeit')
    rfb = pd.melt(rfb, id_vars=['Pfad'], var_name='Zeit')

    # Convert type of 'Zeit' column
    output['Zeit'] = output['Zeit'].astype('int32')
    rfb['Zeit'] = rfb['Zeit'].astype('int32')

    output.sort_values(by=['Pfad','Zeit'], inplace=True)
    rfb.sort_values(by=['Pfad','Zeit'], inplace=True)

    output = output['value'].to_numpy()
    rfb = rfb['value'].to_numpy()

    # Remove timestep 60 as the outputs only go to 59
    input = input[input['Zeit'] != 60]
    # Filter parameters
    parameters = ['Diskontfunktion','Aktien','Dividenden','Immobilien','Mieten','10j Spotrate fuer ZZR','1','3','5','10','15','20','30']
    input = input.loc[:, parameters]

    input = input.to_numpy()

    # Scale inputs to (0,1)
    scaler_input = MinMaxScaler()
    # Scale outputs to (-1,1)
    scaler_output = MinMaxScaler(feature_range = (-1,1))

    input = scaler_input.fit_transform(input)
    rfb_scaled = scaler_input.fit_transform(rfb.reshape(-1,1))
    output = scaler_output.fit_transform(output.reshape(-1,1))

    # Create input data. Take current input parameters along with TIMESTEPS previous input parameters
    # to predict current output.
    features = []
    labels = []

    # TODO: Add previous target values as features
    for i in range(1, len(output)):

        # Predictions can only be made starting at timestep 1
        if i % 60 == 0:
            continue
        if i % 60 == 1:
            # Add padding at timestep 0
            # features_0 = np.concatenate([input[i-1, :], rfb_scaled[i-1], output[i-1]])
            features_0 = np.concatenate([input[i-1, :], output[i-1]])
            window_features = np.array([features_0, features_0])
            features.append(window_features)
            labels.append(output[i])

        else:
            # TODO: Eleganter loesen!
            # features_1 = np.concatenate([input[i - recurrent_timesteps, :], rfb_scaled[i - recurrent_timesteps], output[i - recurrent_timesteps]])
            # features_2 = np.concatenate([input[i - 1, :], rfb_scaled[i - 1], output[i - 1]])
            features_1 = np.concatenate([input[i - recurrent_timesteps, :], output[i - recurrent_timesteps]])
            features_2 = np.concatenate([input[i - 1, :], output[i - 1]])
            window_features = np.array([features_1, features_2])
            features.append(window_features)
            # features.append(input[i - recurrent_timesteps : i, :])
            labels.append(output[i])

        
    # Convert to numpy array and reshape
    features, labels = np.array(features), np.array(labels)

    print('Shape of features:')
    print(features.shape)
    print('Shape of labels:')
    print(labels.shape)

    # TODO: Stratified shuffle: In Trainings- und Testdaten muessen repraesentativ fuer Datensatz sein.
    #       D.h. vor allem im Testset sollten 20% (bei train_ratio = 0.8) aller Scenarien zu allen Zeitschritten
    #       (1-59) enthalten sein.
    if shuffle_data == True:
        features, labels = shuffle(features, labels, random_state=config.RANDOM_SEED)
        

    # Split into train and test sets
    X_train, y_train, X_test, y_test = train_test_split(features, labels, train_ratio)

    return X_train, y_train, X_test, y_test, scaler_output


def train_test_split(features, labels, train_ratio):
    """
    Splits the full dataset (features and labels) into training and test set. Here, it is crucial that
    the split point is not in the middle of a scenario but exactly between two scenarios.
    """

    total_scenarios = len(labels) / 59
    print('total_scenarios: ', total_scenarios)

    train_scenarios = int(total_scenarios * train_ratio)
    print('Number of scenarios in training data: ', train_scenarios)

    idx_train_end = train_scenarios * 59
    print('Split data at index: ', idx_train_end)

    X_train = features[:idx_train_end,:,:]
    y_train = labels[:idx_train_end,:]
    X_test = features[idx_train_end:,:,:]
    y_test = labels[idx_train_end:,:]

    return X_train, y_train, X_test, y_test
