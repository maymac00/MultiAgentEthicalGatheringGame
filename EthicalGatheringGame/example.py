
import gym
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from MultiAgentEthicalGathering import MAEGG


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


config_dict = MAEGG.read_preset("large")
env = MAEGG(**config_dict)
acc_reward = [0] * env.n_agents

env.track = True
env.stash_runs = True
env.reset()
for r in range(50):
    env.reset()
    acc_reward = [0] * env.n_agents
    for i in range(env.max_steps):
        env.getObservation()

        actions = []
        for ag in env.agents.values():
            actions.append(greedy_agent(env.map.current_state, ag, env))

        obs, reward, done, info = env.step(actions)
        acc_reward += reward
        # env.render()
    print(acc_reward)
env.plot_results("median")
pass