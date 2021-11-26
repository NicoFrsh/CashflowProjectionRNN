# Data postprocessing functions
import numpy as np

def recursive_prediction(test_data):
    # TODO: Implementieren
    return True

# Function to compute the average of the output value (over all scenarios) for all timesteps.
def calculate_mean(outputs, projection_time):
    
    mean_outputs = []


    for i in range(0, projection_time - 1):

        # Extract output for each scenario at year i
        outputs_extract = outputs[i::projection_time - 1].copy()

        assert (len(outputs_extract) == len(outputs)/59), 'Step size is not aligned with output length!'

        mean_value = np.mean(outputs_extract)
        mean_outputs.append(mean_value)

    mean_outputs = np.array(mean_outputs)
    return mean_outputs

def calculate_loss(targets, predictions):
    """ 
    Calculates the mean-squared-error of the predictions in relation to the expected target values. 
    To get a meaningful value, the loss is computed by taking the means ...
    """
    targets_mean = calculate_mean(targets, 60)
    predictions_mean = calculate_mean(predictions, 60)
    number_scenarios = len(targets) / 59

    loss = []

    for i in range(len(targets_mean)):

        mse = (targets_mean[i] - predictions_mean[i])**2 
        # average over all scenarios
        mse = mse / number_scenarios
        loss.append(mse)

    return loss

