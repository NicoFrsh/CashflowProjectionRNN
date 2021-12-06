# Configuration parameters
RANDOM_SEED = 1

PATH_SCENARIO = r'/Users/nicofrisch/Dokumente/Python/CashflowProjectionRNN/data/scenario_5000.csv'
PATH_OUTPUT = r'/Users/nicofrisch/Dokumente/Python/CashflowProjectionRNN/data/OutputInflowRfB.csv'
OUTPUT_VARIABLE = 'net profit'
ADDITIONAL_INPUT = 'RfB'

TRAIN_RATIO = 0.8

TIMESTEPS = 2

LSTM_CELLS = 32
LSTM_LAYERS = 1
BATCH_SIZE = 1024
OUTPUT_ACTIVATION = 'tanh'

EPOCHS = 150

VERBOSE = 0
