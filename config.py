# Configuration parameters
RANDOM_SEED = 1

PATH_SCENARIO = r'/Users/nicofrisch/Dokumente/Python/CashflowProjectionRNN/data/scenario_5000.csv'
PATH_OUTPUT = r'/Users/nicofrisch/Dokumente/Python/CashflowProjectionRNN/data/output_5001.csv'
OUTPUT_VARIABLE = 'net profit'
USE_ADDITIONAL_INPUT = False
ADDITIONAL_INPUT = 'gross surplus'

TRAIN_RATIO = 0.8

TIMESTEPS = 2

LSTM_CELLS = 32
LSTM_LAYERS = 1
BATCH_SIZE = 1024
OUTPUT_ACTIVATION = 'tanh'
ADDITIONAL_OUTPUT_ACTIVATION = 'sigmoid'

EPOCHS = 150

VERBOSE = 0

## Parameters for comparison of different versions
# whether the scenario inputs Diskontfunktion, Aktien and Immobilien shall be converted to yearly values
# (True) or left as accum ulated values (False)
use_yearly_inputs = False