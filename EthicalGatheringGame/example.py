import copy
import random

import gym
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from EthicalGatheringGame import MAEGG
from EthicalGatheringGame.presets import tiny, small, medium, large
from EthicalGatheringGame.wrappers import NormalizeReward


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


n = 8
preset = copy.copy(large)
preset["we"] = [1, 99]
preset["donation_capacity"] = 100
preset["max_steps"] = 500
preset["n_agents"] = n
# preset["efficiency"] = [0.1] * n # Equally low efficiency
preset["efficiency"] = [0.5] * n  # Equally medium efficiency
# preset["efficiency"] = [0.85] * n  # Equally high efficiency
# preset["efficiency"] = [i/n for i in range(n)]  # Range efficiency
# preset["efficiency"] = [0.85]*2 + [0.1]*6  # 2 high vs 6 low
# preset["efficiency"] = [0.85]*1 + [0.50]*1 + [0.1]*6  # 1 high vs 1 med vs 1 high

preset["color_by_efficiency"] = True
env = MAEGG(**preset)
env.track = True
env.stash_runs = True

acc_reward = [0] * env.n_agents

env.reset()
for r in range(3):
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
        obs, reward, done, info = env.step(actions)
        acc_reward += reward
        # print(reward)
        # env.render()

print("Mean apples geneated: ", np.mean(env.apple_gen_statistic))
# Print ratio gathered/gather_tries for each agent
for ag in list(env.agents.values()):
    print(
        f"Agent {ag.id} efficiency {ag.efficiency}: obtained {round(ag.gathered / (ag.gather_tries + 1e-10), 2)} of {ag.gather_tries}")

h = copy.deepcopy(env.history)
env.setHistory(h)
env.plot_results("median")
env.get_results()
env.print_results()
pass
