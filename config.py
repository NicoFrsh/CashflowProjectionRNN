# Configuration parameters
from tkinter.tix import Tree


RANDOM_SEED = 1

PATH_SCENARIO = r'/Users/nicofrisch/Dokumente/Python/CashflowProjectionRNN/data/scenario_5000.csv'
PATH_OUTPUT = r'/Users/nicofrisch/Dokumente/Python/CashflowProjectionRNN/data/output_5001.csv'
OUTPUT_VARIABLE = 'net profit'
USE_ADDITIONAL_INPUT = False
ADDITIONAL_INPUT = 'gross surplus'

MODEL_TYPE = 'lstm'
TIMESTEPS = 2

LSTM_CELLS = 32
LSTM_LAYERS = 1
BATCH_SIZE = 1024
# TODO: lineare Aktivierung ausprobieren!
OUTPUT_ACTIVATION = 'tanh'
# sigmoid am besten, da die Inputs alle positiv sind und dann die Skalierung auf (0,1) optimal ist.
ADDITIONAL_OUTPUT_ACTIVATION = 'sigmoid'

TRAIN_RATIO = 0.8
EPOCHS = 500

## Parameters for comparison of different versions
# whether the scenario inputs Diskontfunktion, Aktien and Immobilien shall be converted to yearly values
# (True) or left as accumulated values (False)
# TODO: True testen! Sollte eigentlich besser sein!
use_yearly_inputs = False
# Whether the output (net profit) shall be discounted (True) or not (False)
use_discounted_np = True