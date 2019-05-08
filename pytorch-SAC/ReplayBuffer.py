import random
import numpy as np


class ReplayBuffer(object):
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        # Need to append some garbage value first
        # for python to allocate the space so we can
        # place our trajectory at ptr
        if len(self) < self.size:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        buf = np.array(self.buffer)
        info = np.array([self.size, self.ptr])
        np.save(f"{path}_buffer.npy", buf)
        np.save(f"{path}_info.npy", info)

    def load(self, path):
        self.buffer = np.load(f"{path}_buffer.npy").tolist()
        self.size, self.ptr = np.load(f"{path}_info.npy").tolist()
