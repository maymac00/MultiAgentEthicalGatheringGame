import copy
import random

import gymnasium as gym
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from EthicalGatheringGame.wrappers import NormalizeReward, StatTracker


def greedy_agent(grid, agent, env):
    def find_agent_and_targets(grid):
        target_positions = []
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == '@':
                    target_positions.append((i, j))
        return target_positions

    def closest_target(agent_pos, target_positions):
        min_distance = float('inf')
        closest_target_pos = None
        for target in target_positions:
            distance = abs(agent_pos[0] - target[0]) + abs(agent_pos[1] - target[1])
            if distance < min_distance:
                min_distance = distance
                closest_target_pos = target
        return closest_target_pos

    def greedy_move(agent_pos, target_pos):
        move = 4
        if agent_pos[0] < target_pos[0]:
            move = 1
        elif agent_pos[0] > target_pos[0]:
            move = 0
        elif agent_pos[1] < target_pos[1]:
            move = 3
        elif agent_pos[1] > target_pos[1]:
            move = 2
        return move

    if agent.apples > env.survival_threshold and env.donation_box < env.donation_capacity:
        return 5
    if agent.apples < env.survival_threshold and env.donation_box > 0:
        return 6

    target_positions = find_agent_and_targets(grid)
    target_pos = closest_target(agent.position, target_positions)
    if target_pos is None:
        return 4
    move = greedy_move(agent.position, target_pos)
    return move


n = 5
preset = copy.copy(large)
preset["we"] = [1, 10]
preset["donation_capacity"] = 1000
preset["max_steps"] = 500
preset["n_agents"] = n
# preset["efficiency"] = [0.1] * n # Equally low efficiency
# preset["efficiency"] = [0.5] * n  # Equally medium efficiency
preset["efficiency"] = [0.85, 0.85, 0.2, 0.2, 0.2]  # 40% efficiency
# preset["reward_mode"] = "vectorial"

preset["color_by_efficiency"] = True
env = MAEGG(**preset)
env = StatTracker(env)

env.toggleTrack(True)
env.toggleStash(True)

acc_reward = [0] * env.n_agents

env.reset()
for r in range(50):
    obs, _ = env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        actions_agent = []
        agents = list(env.agents.values())
        random.shuffle(agents)
        untie_prio = np.random.permutation(env.n_agents)
        for ag, _ in sorted(zip(agents, untie_prio), reverse=True, key=lambda x: (x[0].efficiency, x[1])):
            actions_agent.append((greedy_agent(env.map.current_state, ag, env), ag.id))
        actions = [a for a, _ in sorted(actions_agent, key=lambda x: x[1])]
        obs, reward, terminated, truncated, info = env.step(actions)
        acc_reward += reward
        # print(reward)
        env.render(pause=0.2)

h = copy.deepcopy(env.history)
env.setHistory(h)
env.print_results()
env.plot_results("median")
#env.get_results()

pass
