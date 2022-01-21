# Data preprocessing
from curses import use_default_colors
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import config

def prepare_data(scenario_path, outputs_path, output_variable, recurrent_timesteps = config.TIMESTEPS, shuffle_data = False, train_ratio = config.TRAIN_RATIO):
    """
    Prepares data and gets it ready to serve as input to LSTM-Neural-Network.
    Parameters:
        - scenario_path: file path for the scenario file containing all parameters that were used to create the output
        - outputs_path: path to output created by the CFP-Tool
        - output_variable: name of the output that we want to predict with our model
        - recurrent_timesteps: how many previous timesteps are included in the prediction for the current output
        - shuffle_data: whether the data shall be shuffled before training
        - train_ratio: ratio of how much of the data is used as training data
    """

    input = pd.read_csv(scenario_path, skiprows=6)
    output = pd.read_csv(outputs_path)

    # Preprocess data
    additional_input = output[output['Variable'] == config.ADDITIONAL_INPUT]
    output = output[output['Variable'] == output_variable]
    output = output.iloc[:, 0:62]
    additional_input = additional_input.iloc[:, 0:62]
    # Remove 'Variable' column
    output = output.drop(columns=['Variable'])
    additional_input = additional_input.drop(columns=['Variable'])
    # Shift Simulation by -1 to match data from input dataframe.
    output['Simulation'] -= 1
    additional_input['Simulation'] -= 1
    output = output.rename(columns={'Simulation':'Pfad'})
    additional_input = additional_input.rename(columns={'Simulation':'Pfad'})

    # Change format of dataframe
    output = pd.melt(output, id_vars=['Pfad'], var_name='Zeit')
    additional_input = pd.melt(additional_input, id_vars=['Pfad'], var_name='Zeit')

    # Convert type of 'Zeit' column
    output['Zeit'] = output['Zeit'].astype('int32')
    additional_input['Zeit'] = additional_input['Zeit'].astype('int32')

    output.sort_values(by=['Pfad','Zeit'], inplace=True)
    additional_input.sort_values(by=['Pfad','Zeit'], inplace=True)

    output = output['value'].to_numpy()
    additional_input = additional_input['value'].to_numpy()
    # Duplicate first entry of additional_input for padding reasons (additional_input is shifted by 1 compared to scenario input!)
    additional_input = np.insert(additional_input, 0, additional_input[0])

    # Remove timestep 60 as the outputs only go to 59
    input = input[input['Zeit'] != 60]
    # Filter parameters
    parameters = ['Diskontfunktion','Aktien','Dividenden','Immobilien','Mieten','10j Spotrate fuer ZZR','1','3','5','10','15','20','30']
    input = input.loc[:, parameters]

    input = input.to_numpy()

    # Concatenate inputs and additional inputs
    if config.USE_ADDITIONAL_INPUT:
        input = np.concatenate( (input, np.reshape(additional_input[:-1], (-1,1))), axis=1 )

    # Remove first (padded) entry of additional_input to retrieve original array
    additional_input = additional_input[1:]

    # TODO: Erst Train-Test-Split und dann MinMaxScaler!

    # Scale inputs to (0,1)
    # TODO: Additional input scaled to (0,1) as input but scaled to (-1,1) as output?!
    scaler_input = MinMaxScaler()
    # Scale outputs to (-1,1)
    scaler_output = MinMaxScaler(feature_range = (-1,1))
    scaler_additional_input = MinMaxScaler()

    input = scaler_input.fit_transform(input) 
    output = scaler_output.fit_transform(output.reshape(-1,1))
    additional_input = scaler_additional_input.fit_transform(additional_input.reshape(-1,1))

    # Create input data. Take current input parameters along with TIMESTEPS previous input parameters
    # to predict current output.
    features = []
    labels = []
    additional_labels = []

    for i in range(1, len(output)):

        # Predictions can only be made starting at timestep 1
        if i % 60 == 0: # t = 0
            continue

        if i % 60 == 1: # t = 1
            # Add inputs at timesteps 1 and 0 (0 and 0 (padding) for additional_input!)
            features.append(input[i-1 : i+1, :])
            # Set output at timestep 1 as label
            labels.append(output[i])

            if config.USE_ADDITIONAL_INPUT: # Set additional_input at timestep 1 as additional labels of the network
                additional_labels.append(additional_input[i])


        else: # t >= 2
            # Add inputs at timesteps t and t-1 (t-1 and t-2 (padding) for additional_input!)
            features.append(input[i - 1 : i + 1, :])          
            # Set output at timestep t as label
            labels.append(output[i])

            if config.USE_ADDITIONAL_INPUT: # Set additional_input at timestep t as additional labels of the network
                additional_labels.append(additional_input[i])

        
    # Convert to numpy array and reshape
    features, labels, additional_labels = np.array(features), np.array(labels), np.array(additional_labels)

    # TODO: Stratified shuffle: In Trainings- und Testdaten muessen repraesentativ fuer Datensatz sein.
    #       D.h. vor allem im Testset sollten 20% (bei train_ratio = 0.8) aller Scenarien zu allen Zeitschritten
    #       (1-59) enthalten sein.
    if shuffle_data == True:
        features, labels = shuffle(features, labels, additional_labels, random_state=config.RANDOM_SEED)
        

    # Split into train and test sets
    if config.USE_ADDITIONAL_INPUT:
        X_train, y_train, y_2_train, X_test, y_test, y_2_test = train_test_split(features, labels, additional_labels, train_ratio)
        
        return X_train, y_train, y_2_train, X_test, y_test, y_2_test, scaler_output, scaler_additional_input, scaler_input

    else:
        X_train, y_train, X_test, y_test = train_test_split(features, labels, additional_labels, train_ratio)

        return X_train, y_train, X_test, y_test, scaler_output, scaler_input


def train_test_split(features, labels, additional_labels, train_ratio):
    """
    Splits the full dataset (features and labels) into training and test set. Here, it is crucial that
    the split point is not in the middle of a scenario but exactly between two scenarios.
    """

    total_scenarios = len(labels) / 59

    train_scenarios = int(total_scenarios * train_ratio)

    idx_train_end = train_scenarios * 59

    X_train = features[:idx_train_end,:,:]
    y_train = labels[:idx_train_end,:]
    X_test = features[idx_train_end:,:,:]
    y_test = labels[idx_train_end:,:]

    if config.USE_ADDITIONAL_INPUT:
        y_2_train = additional_labels[:idx_train_end,:]
        y_2_test = additional_labels[idx_train_end:,:]

        return X_train, y_train, y_2_train, X_test, y_test, y_2_test

    else:
        return X_train, y_train, X_test, y_test