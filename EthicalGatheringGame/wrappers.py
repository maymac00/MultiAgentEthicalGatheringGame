from gym.wrappers.normalize import RunningMeanStd

from EthicalGatheringGame.MultiAgentEthicalGathering import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large

"""
This file contains different wrappers around the MAEGG RL environment using Gym interface.
"""

import gym
import numpy as np


class DummyPeers(gym.Wrapper):
    """
    This wrapper replaces the peers' actions with a dummy action.
    The dummy mask is a list of booleans, where True means that the agent's action is replaced with a dummy action.
    """

    def __init__(self, env, dummy_mask):
        super().__init__(env)
        self.env = env
        self.dummy_mask = dummy_mask
        if len(dummy_mask) != self.env.n_agents:
            raise ValueError("Dummy mask must be of length n_agents")

    def step(self, action):
        for i in range(len(action)):
            if self.dummy_mask[i]:
                action[i] = MAEGG.STAY
        return super().step(action)


class NormalizeReward(gym.core.Wrapper):
    """
    This class wraps normalize reward from gym to work with multiple agents.
    """

    def __init__(self, env, epsilon=1e-8, gamma=0.99):
        super().__init__(env)
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.active = True

        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(env.n_agents)

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        if self.active:
            self.returns = self.returns * self.gamma + rews
            rews = self.normalize(rews)
            self.returns[dones] = 0.0
        return obs, rews, dones, infos

    def normalize(self, rews):
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)


if __name__ == "__main__":
    """
    Wrapper testing
    """
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    # env = DummyPeers(env)
    env = NormalizeReward(env)
    env.we = [1, 999]
    env.reset()
    for i in range(100):
        obs, rews, _, _ = env.step([np.random.randint(0, 5) for _ in range(env.n_agents)])
        print(rews)
    pass
