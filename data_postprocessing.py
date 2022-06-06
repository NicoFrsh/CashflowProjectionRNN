# Data postprocessing functions
from distutils.file_util import copy_file
import numpy as np
from numpy.core.fromnumeric import shape
import config

# TODO: Check difference between two versions! Liegt an batch_size!
def recursive_prediction_old(X_test, rnn_model):

    num_features = X_test.shape[2]
    
    # y_hat collects all predictions made by our network
    y_hat = []

    for i in range(100):

        if i % config.PROJECTION_TIME == 0: # (t = 1): Take actual net profit from timestep 0 for both input vectors (padding!)
            # Predict for timestep 1
            # y_hat_i = rnn_model.predict(np.reshape(X_test[i,:,:], (-1,2,num_features)))
            y_hat_i = rnn_model(np.reshape(X_test[i,:,:], (-1,2,num_features)))

        elif i % config.PROJECTION_TIME == 1: # (i.e. t = 2): Take actual net profit from timestep 0 for the first input vector
            feature = X_test[i,:,:] # Keep net profit
            feature[1,-1] = y_hat[i-1] # Replace net profit with predicted net profit
            print('FEATURE (t=2):')
            print(feature)
            feature = np.reshape(feature, (-1,2,num_features))
            print('Feature shape: ', feature.shape)
            # Predict for timestep 2
            # y_hat_i = rnn_model.predict(np.reshape(feature, (-1,2,num_features)))
            y_hat_i = rnn_model(feature)
            print('y_hat_i: ', y_hat_i)
        else: # Use previous predictions of net profit as input for both input vectors
            feature = X_test[i,:,:]
            # Replace net profits with predicted net profits
            feature[0,-1] = y_hat[i-2]
            feature[1,-1] = y_hat[i-1]
            # Predict for timestep i > 2
            # y_hat_i = rnn_model.predict(np.reshape(feature, (-1,2,num_features)))
            y_hat_i = rnn_model(np.reshape(feature, (-1,2,num_features)))


        y_hat.append(y_hat_i)

    y_hat = np.array(y_hat)
    y_hat = np.reshape(y_hat, (-1,1))

    return y_hat


def recursive_prediction(X_test, rnn_model, recurrent_timesteps = config.TIMESTEPS):

    num_features = X_test.shape[2]
    projection_time = config.PROJECTION_TIME

    # y_hat collects all predictions made by our network
    y_hat = np.empty((X_test.shape[0], 1))
    y_2_hat = np.empty_like(y_hat) # Prediction of ADDITONAL_INPUT

    for i in range(projection_time):
        # ...
        t_rec = min(recurrent_timesteps + 1, i)

        feature = X_test[i::projection_time, :, :]
        for j in range(1, t_rec + 1):
            # Replace additional input with predictions of additional input
            feature[:,-j,-1] = y_2_hat[i-j::projection_time,0]

        # TODO: Achtung! Pruefen ob das mit batch_size != 1 wirklich funktioniert!
        y_hat_i, y_2_hat_i = rnn_model.predict(feature, batch_size = config.BATCH_SIZE)

        y_hat[i::projection_time] = y_hat_i
        if config.USE_ADDITIONAL_INPUT:
            y_2_hat[i::projection_time] = y_2_hat_i

    if config.USE_ADDITIONAL_INPUT:
        return y_hat, y_2_hat
    else:
        return y_hat


def calculate_mean_per_timestep(outputs, timesteps):
    """
    Calculates the mean output value for all timesteps over all scenarios.
    Returns an array of length <timesteps> containing the mean output value for each timestep.
    """

    mean_outputs = []


    for i in range(timesteps):

        # Extract output for each scenario at year i
        outputs_extract = outputs[i::timesteps].copy()

        assert (len(outputs_extract) == len(outputs)/timesteps), 'Step size is not aligned with output length!'

        mean_value = np.mean(outputs_extract)
        mean_outputs.append(mean_value)

    return np.array(mean_outputs)

def calculate_loss_per_timestep(targets, predictions, timesteps = config.PROJECTION_TIME, loss_metric = 'mse'):
    """ 
    Calculates the loss per timestep over all scenarios. You can choose between Mean-Squared-Error ('mse')
    and Mean-Absolute-Error ('mae') as loss metric.
    Returns an array of length <projection_time> (we start predicting at timestep 1)
    containing the MSE for each timestep.
    """

    loss = []

    for i in range(timesteps):

        # Extract targets and predictions for each scenario at timestep i
        targets_extract = targets[i::timesteps].copy()
        predictions_extract = predictions[i::timesteps].copy()

        assert (len(targets_extract) == len(targets)/timesteps and 
        len(predictions_extract) == len(predictions)/timesteps), 'Step size is not aligned with output length!'            

        if (loss_metric == 'mse'):
            # Compute MSE at timestep i
            loss_i = np.mean((np.square(targets_extract - predictions_extract)))

        elif (loss_metric == 'mae'):
            # Compute MAE at timestep i
            loss_i = (np.abs(targets_extract - predictions_extract)).mean()

        loss.append(loss_i)
            
    # Check:
    # total_loss = np.mean(loss)
    # print('total_loss (per timestep): ', total_loss)
    loss = np.array(loss)
    return loss

def calculate_mean_per_scenario(outputs, timesteps):
    """
    Calculates the mean output value for each scenario over all time steps.
    Returns an array of length <number_scenarios> containing the mean output value for each scenario.
    """

    number_scenarios = int(len(outputs)/timesteps)
    mean_outputs = []

    for i in range(number_scenarios):

        # Extract all timesteps for i-th scenario
        outputs_extract = outputs[i * timesteps : (i+1) * timesteps]

        assert (len(outputs_extract) == timesteps), 'Step size is not aligned with output length!'

        mean_value = np.mean(outputs_extract)
        mean_outputs.append(mean_value)

    mean_outputs = np.array(mean_outputs)

    return mean_outputs

def calculate_loss_per_scenario(targets, predictions, timesteps = config.PROJECTION_TIME, loss_metric = 'mse'):
    """
    Calculates the mean-squared-error per scenario over all timesteps. You can choose between Mean-Squared-Error ('mse')
    and Mean-Absolute-Error ('mae') as loss metric.
    Returns an array of length <number_scenarios> containing the MSE for each scenario.
    """

    loss = []

    number_scenarios = int(len(targets)/timesteps)

    for i in range(number_scenarios):

        targets_extract = targets[i * timesteps : (i+1) * timesteps]
        predictions_extract = predictions[i * timesteps : (i+1) * timesteps]

        assert (len(targets_extract) == timesteps and
        len(predictions_extract) == timesteps), 'Step size is not aligned with output length!'

        if (loss_metric == 'mse'):
            # Compute MSE of i-th scenario
            loss_i = (np.square(targets_extract - predictions_extract)).mean()

        elif (loss_metric == 'mae'):
            # Compute MAE of i-th scenario
            loss_i = (np.abs(targets_extract - predictions_extract)).mean()

        loss.append(loss_i)

    loss = np.array(loss)
    # Check:
    # total_loss = np.mean(loss)
    # print('total_loss (per scenario): ', total_loss)
    return loss

# Function that calculates the stochastic PVFP as:
#               stoch. PVFP = mean of PVFP_s
#       where s are the scenarios.
def calculate_pvfp(net_profits, scenario, discount_functions = None):
    """
    Calculates the PVFP for a specific scenario using the formula
        PVFP = sum_t d(t) * net_profit(t)
    """

    if config.use_discounted_np:
        net_profits = np.array(net_profits).reshape((-1,))
         # Get net profits for specified scenario
        net_profits_s = net_profits[scenario*config.PROJECTION_TIME : (scenario+1)*config.PROJECTION_TIME]

        return np.sum(net_profits_s)

    else:
        net_profits, discount_functions = np.array(net_profits).reshape((-1,)), np.array(discount_functions).reshape((-1,))
        # Get net profits for specified scenario
        net_profits_s = net_profits[scenario*config.PROJECTION_TIME : (scenario+1)*config.PROJECTION_TIME]
        # Get discount function from inputs
        discount_function_s = discount_functions[scenario*config.PROJECTION_TIME : (scenario+1)*config.PROJECTION_TIME]

        return np.dot(net_profits_s, discount_function_s)

def calculate_stochastic_pvfp(net_profits, discount_functions = None):

    net_profits, discount_functions = np.array(net_profits).reshape((-1,)), np.array(discount_functions).reshape((-1,))

    number_scenarios = int(net_profits.size / config.PROJECTION_TIME)
    
    pvfps = [calculate_pvfp(net_profits, discount_functions, scenario) for scenario in range(number_scenarios)]

    return np.mean(np.array(pvfps))

# Alternative (Check if correct!)
def calculate_stoch_pvfp(net_profits, discount_functions = None):

    number_scenarios = net_profits.size / config.PROJECTION_TIME

    if not config.use_discounted_np:
        net_profits, discount_functions = np.array(net_profits).reshape((-1,)), np.array(discount_functions).reshape((-1,))

        pvfps = np.dot(net_profits, discount_functions)

        return pvfps / number_scenarios

    else: 
        return np.sum(net_profits) / number_scenarios

def create_discount_vector(spot_rates, scenario):
    """
    Calculates the discount vector based on the zero-coupon prices for a specific scenario:
        discount_vector[0] = 1.0
        discount_vector[t] = discount_vector[t-1] * zcb_prices[t-1]
    """
    discount_vector = np.zeros(config.PROJECTION_TIME)

    for t in range(config.PROJECTION_TIME):
        if t == 0:
            discount_vector[t] = 1.0
        else:
            # Convert spot rate to zcb price
            zcb_price = 1 / (1 + spot_rates[config.PROJECTION_TIME*scenario + t - 1])
            discount_vector[t] = discount_vector[t-1] * zcb_price

    return discount_vector