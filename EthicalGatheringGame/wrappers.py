from typing import Tuple

from gymnasium.wrappers.normalize import RunningMeanStd
from prettytable import PrettyTable

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

        env.unwrapped.logger.info("Dummy peers wrapper initialized")
        env.unwrapped.logger.info(f"Dummy mask: {dummy_mask}")
        if len(dummy_mask) != self.env.unwrapped.n_agents:
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
        self.returns = np.zeros(env.unwrapped.n_agents)

        env.unwrapped.logger.info("Normalize reward wrapper initialized")
        env.unwrapped.logger.info(f"Gamma: {gamma}")

        if env.unwrapped.reward_mode == "vectorial":
            raise ValueError("Vectorial rewards are not supported.")

    def step(self, action):
        obs, rews, terminated, truncated, infos = self.env.step(action)
        terminated = [terminated] * self.env.unwrapped.n_agents
        if self.active:
            self.returns = self.returns * self.gamma + rews
            rews = self.normalize(rews)
            self.returns[terminated] = 0.0
        return obs, rews, terminated, truncated, infos

    def normalize(self, rews):
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)


class StatTracker(gym.core.Wrapper):
    """
    This class tracks the running mean and std of the main statistics of the agents. Includes:
    - Number of apples gathered
    - Number of apples dropped
    - Number of apples taken from the box
    - Time to survival
    - Apples generated on the map
    """

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.apples_gathered = [RunningMeanStd(shape=()) for _ in range(env.unwrapped.n_agents)]
        self.apples_dropped = [RunningMeanStd(shape=()) for _ in range(env.unwrapped.n_agents)]
        self.apples_from_box = [RunningMeanStd(shape=()) for _ in range(env.unwrapped.n_agents)]
        self.apples_generated = RunningMeanStd(shape=())
        self.global_apples_gather = RunningMeanStd(shape=())
        self.global_apples_dropped = RunningMeanStd(shape=())

    def step(self, action):
        obs, rews, terminated, truncated, info = self.env.step(action)

        agents = self.env.unwrapped.agents

        if terminated or truncated:
            gather = np.array([agent.gathered for agent in agents.values()]).sum()
            drop = np.array([agent.apples_dropped for agent in agents.values()]).sum()
            self.global_apples_gather.update(np.expand_dims([gather], axis=0))
            self.global_apples_dropped.update(np.expand_dims([drop], axis=0))

            for i, agent in agents.items():
                self.apples_gathered[i].update(np.expand_dims([agent.gathered], axis=0))
                self.apples_dropped[i].update(np.expand_dims([agent.apples_dropped], axis=0))
                self.apples_from_box[i].update(np.expand_dims([agent.apples_from_box], axis=0))
            self.apples_generated.update(np.expand_dims([info["sim_data"]["generated_apples"]], axis=0))
        return obs, rews, terminated, truncated, info

    def print_results(self):
        results = self.env.print_results()
        # Plot as fancy table with
        table = PrettyTable()

        title = "Global statistics"
        print("=" * len(title))
        print(title)
        print("=" * len(title))

        print("Mean + std of apples generated on the map: ", round(self.apples_generated.mean[0], 2), "+-",
              round(self.apples_generated.var[0], 2))
        print("Mean + std of apples taken: ", round(self.global_apples_gather.mean[0],2), "+-",
              round(self.global_apples_gather.var[0],2), "from which:")

        title = "Agent statistics"
        print("=" * len(title))
        print(title)
        print("=" * len(title))

        table.field_names = ["Agent", "Apples gathered", "Gathering ratio", "Failed to gather", "Apples from box"]

        for i in range(self.env.unwrapped.n_agents):
            self.apples_gathered[i].mean = self.apples_gathered[i].mean.round(2)
            self.apples_dropped[i].mean = self.apples_dropped[i].mean.round(2)
            self.apples_from_box[i].mean = self.apples_from_box[i].mean.round(2)

            table.add_row([i,
                           self.apples_gathered[i].mean.item(),
                           round((self.apples_gathered[i].mean/ self.apples_generated.mean)[0].item(), 2),
                           self.apples_dropped[i].mean.item(),
                           self.apples_from_box[i].mean.item()])

        print(table)
        results["stats_table"] = np.array(table._rows)
        results["stats_header"] = table.field_names
        return results


if __name__ == "__main__":
    """
    Wrapper testing
    """
    env = gym.make("MultiAgentEthicalGathering-v1", **tiny)
    # env = DummyPeers(env)
    env = NormalizeReward(env)
    env.unwrapped.we = [1, 999]
    env.unwrapped.reset()
    for i in range(100):
        obs, rews, _, _ = env.step([np.random.randint(0, 5) for _ in range(env.unwrapped.n_agents)])
        print(rews)
    pass
