# Data preprocessing
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
import config

def prepare_data(scenario_path, outputs_path, output_variable, projection_time = config.PROJECTION_TIME, 
                recurrent_timesteps = config.TIMESTEPS, output_activation = config.OUTPUT_ACTIVATION, 
                additional_output_activation = config.ADDITIONAL_OUTPUT_ACTIVATION, shuffle_data = config.SHUFFLE, train_ratio = config.TRAIN_RATIO):
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

    # np.random.seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    input = pd.read_csv(scenario_path, skiprows=6)
    output = pd.read_csv(outputs_path)

    # Preprocess data
    additional_input = output[output['Variable'] == config.ADDITIONAL_INPUT]
    output = output[output['Variable'] == output_variable]
    output = output.iloc[:, 0:projection_time+3]
    additional_input = additional_input.iloc[:, 0:projection_time+3]
    # Remove 'Variable' column
    output = output.drop(columns=['Variable'])
    additional_input = additional_input.drop(columns=['Variable'])
    # Rename column
    output = output.rename(columns={'Simulation':'Pfad'})
    additional_input = additional_input.rename(columns={'Simulation':'Pfad'})

    # Remove timestep 61 as the outputs only go to 60
    input = input[input['Zeit'] != projection_time + 1]
    input = input[input['Zeit'] != projection_time + 2]

    # Filter parameters
    parameters = ['Diskontfunktion','Aktien','Dividenden','Immobilien','Mieten','10j Spotrate fuer ZZR','1','3','5','10','15','20','30']
    input = input.loc[:, parameters]

    # Change format of dataframe to long table
    output = pd.melt(output, id_vars=['Pfad'], var_name='Zeit')
    additional_input = pd.melt(additional_input, id_vars=['Pfad'], var_name='Zeit')

    # Convert type of 'Zeit' column
    output['Zeit'] = output['Zeit'].astype('int32')
    additional_input['Zeit'] = additional_input['Zeit'].astype('int32')

    output.sort_values(by=['Pfad','Zeit'], inplace=True)
    additional_input.sort_values(by=['Pfad','Zeit'], inplace=True)

    output = output['value'].to_numpy()
    additional_input = additional_input['value'].to_numpy()
    # Create copy of additional_input for additional_input labels (no padding etc)
    additional_output = additional_input.copy()

    # Adjust discount function column
    discount_vector = generate_discount_vector(input)
    input['Diskontfunktion'] = discount_vector

    # Convert output to discounted output using the discount function of the scenario file
    if config.use_discounted_np:

        discount_function = input.loc[:, 'Diskontfunktion']
        discount_function = discount_function.to_numpy()
        output *= discount_function
        additional_output *= discount_function
        additional_input *= discount_function

    # Duplicate first entry of additional_input for padding reasons (additional_input is shifted by 1 compared to scenario input!)
    indices = np.arange(0, additional_input.size + 1, projection_time + 1)
    # Remove last entry of each scenario (which will not be used to predict the last net profit)
    additional_input = np.delete(additional_input, indices[1:]-1)
    # Add padding for first entry (= 0 for all scenarios) (based on config.TIMESTEPS)
    additional_input = add_padding_to_additional_input(additional_input, recurrent_timesteps, projection_time)

    # Convert Diskontfunktion, Aktien and Immobilien to yearly instead of accumulated values using the formula
    # x_t = (x_t / x_{t-1}) - 1
    if config.use_yearly_inputs:

        # Separate yearly and accumulated inputs
        yearly_inputs = input.loc[:, ['Diskontfunktion','Aktien','Immobilien']]
        input = input.drop(columns=['Diskontfunktion','Aktien','Immobilien'])
        yearly_inputs = yearly_inputs.to_numpy()
        input = input.to_numpy()

        for i in reversed(range(len(yearly_inputs))): 
            
            if i % (projection_time+1) == 0:
                yearly_inputs[i,:] = 0
            else: # Value at t = 0 is always 0
                yearly_inputs[i,:] = (yearly_inputs[i,:] / yearly_inputs[i-1,:]) - 1

        scaler_yearly_inputs = MinMaxScaler(feature_range=(0,1))
        yearly_inputs = scaler_yearly_inputs.fit_transform(yearly_inputs)
        scaler_input = MinMaxScaler(feature_range=(0,1))
        input = scaler_input.fit_transform(input)

        input = np.concatenate((input, yearly_inputs), axis=1)
  
    else:
        input = input.to_numpy()
        scaler_input = MinMaxScaler()
        input = scaler_input.fit_transform(input)


    # Add padding to input
    if recurrent_timesteps > 1:
        input = add_padding_to_input(input, recurrent_timesteps, projection_time)

    print('shape of input: ', input.shape)

    # TODO: Erst Train-Test-Split und dann MinMaxScaler!

    if config.USE_ADDITIONAL_INPUT:
        if (additional_output_activation == 'sigmoid' or additional_output_activation == 'relu'):
            # Scale additional input/output to (0,1) to be conform with the sigmoid activation function whos value lie in (0,1)
            scaler_additional_output = MinMaxScaler()
        elif (additional_output_activation == 'tanh' or additional_output_activation == 'linear'):
            # Scale additional inputs to (-1,1) to be conform with the tanh activation function whos value lie in (-1,1)
            scaler_additional_output = MinMaxScaler(feature_range=(-1,1))

        additional_output = scaler_additional_output.fit_transform(additional_output.reshape(-1,1))
        additional_input = scaler_additional_output.fit_transform(additional_input.reshape(-1,1))    
        
        # Concatenate inputs and additional inputs
        input = np.concatenate( (input, additional_input), axis=1 )

    if (output_activation == 'tanh' or output_activation == 'linear'):
        # Scale outputs to (-1,1)
        scaler_output = MinMaxScaler(feature_range = (-1,1))
    elif (output_activation == 'sigmoid' or output_activation == 'relu'):
        # Scale outputs to (0,1)
        scaler_output = MinMaxScaler()

    output = scaler_output.fit_transform(output.reshape(-1,1))            

    # Create input data. Take current input parameters along with TIMESTEPS previous input parameters
    # to predict current output.
    features = []
    labels = []
    additional_labels = []

    print('len(output): ', len(output))

    # Create labels
    for i in range(1, len(output)):
        # Predictions can only be made starting at timestep 1
        if i % (projection_time + 1) != 0:
            # Set output at timestep t as label
            labels.append(output[i])

            if config.USE_ADDITIONAL_INPUT: # Set additional_input at timestep t as additional labels of the network
                additional_labels.append(additional_output[i])

    # Create features
    for i in range(input.shape[0]):
        # Add inputs at timesteps t,t-1,...,t-recurrent_timesteps (t-1,t-2,...,t-(recurrent_timesteps+1) for additional_input!)
        if i % (projection_time + recurrent_timesteps) > recurrent_timesteps - 1:
            features.append(input[i - recurrent_timesteps : i + 1, :])
    
    # Convert to numpy array and reshape
    features, labels, additional_labels = np.array(features), np.array(labels), np.array(additional_labels)

    print('feature shape: ', features.shape)
    print('labels shape: ', labels.shape)
    print('add_labels shape: ', additional_labels.shape)

    if shuffle_data:

        features_batches, labels_batches, additional_labels_batches = [],[],[]
        number_scenarios = int(labels.shape[0] / projection_time)

        for s in range(number_scenarios):
            features_batches.append(features[s*projection_time : (s+1)*projection_time,:,:])
            labels_batches.append(labels[s*projection_time : (s+1)*projection_time, : ]) 
            additional_labels_batches.append(additional_labels[s*projection_time : (s+1)*projection_time, : ])

        # Shuffle batch-wise, so that features and labels remain aligned
        features, labels, additional_labels = shuffle(features_batches, labels_batches, additional_labels_batches, random_state=config.RANDOM_SEED)
        features, labels, additional_labels = np.array(features), np.array(labels), np.array(additional_labels)

        # Reshape arrays
        features = features.reshape(-1, recurrent_timesteps + 1, features.shape[-1])
        labels = labels.reshape(-1, 1)
        additional_labels = additional_labels.reshape(-1,1)
        

    # Split into train, validation and test sets
    if config.USE_ADDITIONAL_INPUT:
        X_train, y_train, y_2_train, X_val, y_val, y_2_val, X_test, y_test, y_2_test = train_val_test_split(features, labels, additional_labels, train_ratio)
        return X_train, y_train, y_2_train, X_val, y_val, y_2_val, X_test, y_test, y_2_test, scaler_output, scaler_additional_output, scaler_input

    else:
        X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(features, labels, additional_labels, train_ratio)
        return X_train, y_train, X_val, y_val, X_test, y_test, scaler_output, scaler_input


def train_val_test_split(features, labels, additional_labels, train_ratio, projection_time = config.PROJECTION_TIME):
    """
    Splits the full dataset (features and labels) into training, validation and test set using 'train_ratio' percent of
    the data as training data and the rest evenly distributed as validation and test data. E.g. train_ratio = 0.9:  
            0.9 : 0.05 : 0.05 - Split
    Here, it is crucial that the split point is not in the middle of a scenario but exactly between two scenarios.
    """

    total_scenarios = len(labels) / projection_time

    train_scenarios_end = int(total_scenarios * train_ratio)
    val_scenarios_end = int(total_scenarios * (train_ratio + ((1-train_ratio)/2)))

    idx_train_end = train_scenarios_end * projection_time
    idx_val_end = val_scenarios_end * projection_time

    X_train = features[:idx_train_end,:,:]
    y_train = labels[:idx_train_end,:]
    X_val = features[idx_train_end:idx_val_end,:,:]
    y_val = labels[idx_train_end:idx_val_end,:]
    X_test = features[idx_val_end:,:,:]
    y_test = labels[idx_val_end:,:]


    if config.USE_ADDITIONAL_INPUT:
        y_2_train = additional_labels[:idx_train_end,:]
        y_2_val = additional_labels[idx_train_end:idx_val_end,:]
        y_2_test = additional_labels[idx_val_end:,:]

        return X_train, y_train, y_2_train, X_val, y_val, y_2_val, X_test, y_test, y_2_test

    else:
        return X_train, y_train, X_val, y_val, X_test, y_test

def add_padding_to_additional_input(input, timesteps, projection_time):
    """
    Padding for additional_input. To predict the net profit at timestep t the neural network needs additional_input at timesteps
    t-1, t-2, ..., t-(timesteps+1). So, e.g. to predict net profit at timestep 1, we need padding and use the additional_input at 
    t=0 (timesteps+1) times.
    """
    for i in range(timesteps):

        indices = np.arange(0, input.size + 1, projection_time + i)

        input = np.insert(input, indices[:-1], input[indices[:-1]])

    return input

def add_padding_to_input(input, timesteps, projection_time):
    """
    Padding for regular input. To predict the net profit at timestep t the neural network needs inputs at timesteps
    t, t-1, ..., t-timesteps. So, e.g. to predict net profit at timestep 1, we need padding and use the inputs at 
    t=1, plus t=0 timesteps times.
    """

    for i in range(timesteps - 1):

        indices = np.arange(0, input.shape[0] + 1, projection_time + i + 1)
        # print('indices: ', indices[:10])
        # print('length: ', len(indices))
        # print('shape of input: ', input.shape)
        input = np.insert(input, indices[:-1], input[indices[:-1],:], axis=0)

    return input

def generate_discount_vector(input_df):

    spot_rates = input_df.loc[:,'1']
    spot_rates = spot_rates.to_numpy()
    discount_vector = np.ones(len(input_df))

    for i in range(len(input_df)):

        if i % (config.PROJECTION_TIME+1) != 0:
            # print('input_df entry: ', spot_rates[i-1])
            discount_vector[i] = discount_vector[i-1] * (1 / (1 + spot_rates[i-1]))

    return discount_vector

