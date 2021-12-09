# Data postprocessing functions
import numpy as np

def recursive_prediction(X_test, rnn_model):

    # y_hat collects all predictions made by our network
    y_hat = []

    for i in range(len(X_test)):

        if i % 59 == 0: # (t = 1): Take actual net profit from timestep 0 for both input vectors (padding!)
            # Predict for timestep 1
            y_hat_i = rnn_model.predict(np.reshape(X_test[i,:,:], (-1,2,14)))

        elif i % 59 == 1: # (i.e. t = 2): Take actual net profit from timestep 0 for the first input vector
            feature = X_test[i,:,:] # Keep net profit
            feature[1,-1] = y_hat[i-1] # Replace net profit with predicted net profit
            # Predict for timestep 2
            y_hat_i = rnn_model.predict(np.reshape(feature, (-1,2,14)))
        else: # Use previous predictions of net profit as input for both input vectors
            feature = X_test[i,:,:]
            # Replace net profits with predicted net profits
            feature[0,-1] = y_hat[i-2]
            feature[1,-1] = y_hat[i-1]
            # Predict for timestep i > 2
            y_hat_i = rnn_model.predict(np.reshape(feature, (-1,2,14)))

        y_hat.append(y_hat_i)

    y_hat = np.array(y_hat)
    y_hat = np.reshape(y_hat, (-1,1))

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

    mean_outputs = np.array(mean_outputs)

    return mean_outputs

def calculate_loss_per_timestep(targets, predictions, timesteps = 59, loss_metric = 'mse'):
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
    total_loss = np.mean(loss)
    print('total_loss (per timestep): ', total_loss)
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

def calculate_loss_per_scenario(targets, predictions, timesteps = 59, loss_metric = 'mse'):
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
    total_loss = np.mean(loss)
    print('total_loss (per scenario): ', total_loss)
    return loss

# def calculate_loss_per_scenario(targets, predictions, loss_metric = 'mse'):
#     """
#     Calculates the mean-squared-error per scenario over all timesteps. You can choose between Mean-Squared-Error ('mse')
#     and Mean-Absolute-Error ('mae') as loss metric.
#     Returns an array of length <number_scenarios> containing the MSE for each scenario.
#     """
#     targets_mean = calculate_mean_per_scenario(targets, 59)
#     predictions_mean = calculate_mean_per_scenario(predictions, 59)
#     # TODO: Problem: Anzahl Szenarien ist nicht immer 5001!! Training: 5001*0.8 z.b.
#     number_timesteps = 59

#     loss = []

#     for i in range(len(targets_mean)):

#         if loss_metric == 'mse':
#             current_loss = (targets_mean[i] - predictions_mean[i])#**2
#             # current_loss = current_loss / number_timesteps

#         elif loss_metric == 'mae':
#             current_loss = abs(targets_mean[i] - predictions_mean[i])
#             # current_loss = current_loss / number_timesteps
        
#         loss.append(current_loss)

#     # Check if total loss is same as the score returned by evaluate function
#     total_loss = np.sum(np.array(loss))
#     print('total loss (per scenario), {}: '.format(loss_metric), total_loss)

#     return loss
