import torch

LEARNING_RATE = 3e-4
GAMMA = 0.99
INITIAL_REPLAY_SIZE = 10000
REPLAY_SIZE = 1000000
REPLAY_TYPE = 'default'     # default or combined
PLAY_STEPS = 1
HIDDEN_SIZE = 256
BATCH_SIZE = 256
TAU = 0.005
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LOGSTD_MIN = -20
LOGSTD_MAX = 2
EVAL_FREQ = 10000
NUM_EVAL_GAMES = 10
RENDER_EVAL_FREQ = 50000
SUMMARY_FREQ = 1000
SAVE_FREQ = 50000
NUM_ITERATIONS = 5000000
MAX_STEPS = 1000
