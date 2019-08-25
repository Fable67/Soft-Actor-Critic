import torch
from ReplayBuffer import ReplayBuffer
from CombinedReplayBuffer import CombinedReplayBuffer
import torch.optim as optim
from ranger import Ranger


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

POLICY_LEARNING_RATE = 3e-4
Q_LEARNING_RATE = 3e-4
ALPHA_LEARNING_RATE = 3e-4
POLICY_OPTIM = optim.Adam   # Ranger
Q_OPTIM = optim.Adam		# Ranger
ALPHA_OPTIM = optim.Adam    # Ranger

GAMMA = 0.99
TAU = 0.005
LOGSTD_MIN = -20
LOGSTD_MAX = 2
GRADIENT_STEPS = 1

INITIAL_REPLAY_SIZE = 10000
REPLAY_SIZE = 1000000
REPLAY_BUFFER = ReplayBuffer

HIDDEN_SIZE = 256
BATCH_SIZE = 256

NUM_ITERATIONS = 5000000
EVAL_FREQ = 10000
NUM_EVAL_GAMES = 10
RENDER_EVAL_FREQ = 50000
SUMMARY_FREQ = 1000
SAVE_FREQ = 50000
MAX_STEPS = 1000
