import gym
import numpy as np


class NormalizedActions(gym.ActionWrapper):
    def _action(self, action):
        low = self.action_space.low
        high = self.action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)
        action = np.clip(action, low, high)
        return action

    def _reverse_action(self, action):
        low = self.action_space.low
        high = self.action_space.high
        action = action / (high - low) * 2 - low - 1
        action = np.clip(action, low, high)
        return action
