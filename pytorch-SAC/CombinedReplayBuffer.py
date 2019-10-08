import random
import numpy as np
import math

# https://www.endtoend.ai/slowpapers/a-deeper-look-at-experience-replay/


class CombinedReplayBuffer(object):
    def __init__(self, size, short_term_memory_percentage=0.1):
        self.size = size
        self.short_size = round(size * short_term_memory_percentage)
        self.buffer = []
        self.short_term_buffer_idxs = []
        self.ptr = 0
        self.short_ptr = 0

    def add(self, state, action, reward, next_state, done):
        if len(self) < self.size:
            self.buffer.append(None)
        if len(self.short_term_buffer_idxs) < self.short_size:
            self.short_term_buffer_idxs.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.short_term_buffer_idxs[self.short_ptr] = self.ptr
        self.ptr = (self.ptr + 1) % self.size
        self.short_ptr = (self.short_ptr + 1) % self.short_size

    def sample(self, batch_size, short_term_proportion=0.5):
        batch = [self.buffer[idx] for idx in random.sample(
            self.short_term_buffer_idxs, math.floor(batch_size * short_term_proportion))]
        batch += random.sample(self.buffer, math.ceil(batch_size * short_term_proportion))
        s, a, r, ns, d = map(np.array, zip(*batch))
        return s, a, r, ns, d

    def __len__(self):
        return len(self.buffer)

    def save(self, path):
        buf = np.array([self.buffer, self.short_term_buffer_idxs])
        info = np.array([self.size, self.short_size, self.ptr, self.short_ptr])
        np.save(f"{path}_buffer.npy", buf)
        np.save(f"{path}_info.npy", info)

    def load(self, path):
        self.buffer, self.short_term_buffer_idxs = np.load(f"{path}_buffer.npy").tolist()
        self.size, self.short_size, self.ptr, self.short_ptr = np.load(f"{path}_info.npy").tolist()
