# Data postprocessing functions
import numpy as np

def recursive_prediction(test_data):
    # TODO: Implementieren
    
    return True

# Function to compute the average of the output value (over all scenarios) for all timesteps.
def calculate_mean_per_timestep(outputs, timesteps):
    
    mean_outputs = []


    for i in range(timesteps):

        # Extract output for each scenario at year i
        outputs_extract = outputs[i::timesteps].copy()

        assert (len(outputs_extract) == len(outputs)/timesteps), 'Step size is not aligned with output length!'

        mean_value = np.mean(outputs_extract)
        mean_outputs.append(mean_value)

    mean_outputs = np.array(mean_outputs)

    return mean_outputs

def calculate_mean_per_scenario(outputs, timesteps):

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

def calculate_loss_per_timestep(targets, predictions, loss_metric = 'mse'):
    """ 
    Calculates the loss per timestep over all scenarios. You can chosse between Mean-Squared-Error ('mse')
    and Mean-Absolute-Error ('mae') as loss metric.
    Returns an array of length <projection_time - 1> (we start predicting at timestep 1)
    containing the MSE for each timestep.
    """
    targets_mean = calculate_mean_per_timestep(targets, 59)
    predictions_mean = calculate_mean_per_timestep(predictions, 59)
    number_scenarios = len(targets) / 59

    loss = []

    for i in range(len(targets_mean)):

        if loss_metric == 'mse':
            current_loss = (targets_mean[i] - predictions_mean[i])**2 
            current_loss = current_loss / number_scenarios

        elif loss_metric == 'mae':
            current_loss = abs(targets_mean[i] - predictions_mean[i])
            current_loss = current_loss / number_scenarios
        
        loss.append(current_loss)

    return loss

def calculate_loss_per_scenario(targets, predictions, loss_metric = 'mse'):
    """
    Calculates the mean-squared-error per scenario over all timesteps. You can chosse between Mean-Squared-Error ('mse')
    and Mean-Absolute-Error ('mae') as loss metric.
    Returns an array of length <number_scenarios> containing the MSE for each scenario.
    """
    targets_mean = calculate_mean_per_scenario(targets, 59)
    predictions_mean = calculate_mean_per_scenario(predictions, 59)
    # TODO: Problem: Anzahl Szenarien ist nicht immer 5001!! Training: 5001*0.8 z.b.
    number_timesteps = 59

    loss = []

    for i in range(len(targets_mean)):

        if loss_metric == 'mse':
            current_loss = (targets_mean[i] - predictions_mean[i])**2
            current_loss = current_loss / number_timesteps

        elif loss_metric == 'mae':
            current_loss = abs(targets_mean[i] - predictions_mean[i])
            current_loss = current_loss / number_timesteps
        
        loss.append(current_loss)

    return loss
