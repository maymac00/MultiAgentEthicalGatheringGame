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
        # Set all actions but last to STAY
        for i in range(len(action)):
            if self.dummy_mask[i]:
                action[i] = MAEGG.STAY
        return super().step(action)


if __name__ == "__main__":
    """
    Wrapper testing
    """
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    env = DummyPeers(env)
    env.reset()
    for i in range(10):
        obs, _, _, _ = env.step([np.random.randint(0, 5) for _ in range(env.n_agents)])
        env.render(mode="text")
    pass
