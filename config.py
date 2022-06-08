# Configuration parameters

RANDOM_SEED = 123
SHUFFLE = True

PATH_SCENARIO = r'/Users/nicofrisch/Dokumente/Python/CashflowProjectionRNN/data/scenario_60_10k.csv'
PATH_OUTPUT = r'/Users/nicofrisch/Dokumente/Python/CashflowProjectionRNN/data/output_60_10k.csv'
OUTPUT_VARIABLE = 'net profit'
USE_ADDITIONAL_INPUT = False
ADDITIONAL_INPUT = 'gross surplus'

PROJECTION_TIME = 60
TIMESTEPS = 2

MODEL_TYPE = 'lstm'
LEARNING_RATE = 0.01

LSTM_CELLS = 32
LSTM_LAYERS = 1
BATCH_SIZE = 500
# Note: All activations other than 'tanh' are slower, as CuDNN is only implemented for 'tanh'.
RNN_ACTIVATION = 'tanh'
# TODO: lineare Aktivierung ausprobieren!
OUTPUT_ACTIVATION = 'tanh'
# sigmoid am besten, da die Inputs alle positiv sind und dann die Skalierung auf (0,1) optimal ist.
ADDITIONAL_OUTPUT_ACTIVATION = 'sigmoid'

TRAIN_RATIO = 0.9
EPOCHS = 500

## Parameters for comparison of different versions
# whether the scenario inputs Diskontfunktion, Aktien and Immobilien shall be converted to yearly values
# (True) or left as accumulated values (False)
# TODO: True testen! Sollte eigentlich besser sein!
use_yearly_inputs = True 
# Whether the output (net profit) shall be discounted (True) or not (False)
use_discounted_np = True

# Descriptive name for directory where the model is saved
MODEL_PATH = 'models_60_10k/{}_'.format(MODEL_TYPE)
if SHUFFLE:
    MODEL_PATH += 'shuffle_'
if USE_ADDITIONAL_INPUT:
    MODEL_PATH += str.replace(ADDITIONAL_INPUT, ' ', '_') + '_'
if use_yearly_inputs:
    MODEL_PATH += 'yearly_'
if use_discounted_np:
    MODEL_PATH += 'discounted_'
if RNN_ACTIVATION != 'tanh':
    MODEL_PATH += RNN_ACTIVATION + '_'
if OUTPUT_ACTIVATION != 'tanh':
    MODEL_PATH += OUTPUT_ACTIVATION + '_'
MODEL_PATH += 'T_{0}_bs_{1}_{2}_{3}'.format(TIMESTEPS, BATCH_SIZE, LSTM_LAYERS, LSTM_CELLS)