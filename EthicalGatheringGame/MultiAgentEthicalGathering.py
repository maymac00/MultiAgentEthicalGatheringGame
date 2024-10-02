from collections import deque

import numpy as np

from matplotlib import pyplot as plt

import gymnasium as gym
import json
from EthicalGatheringGame.Maps import Maps
import bisect
import logging
import prettytable
from pettingzoo import ParallelEnv

# TODO: Check env.Wrapper subclassing to achieve: action space mapping, callbacks, last action memory, etc. This will
#  keep the base env simpler
class Agent:
    # Alphabet for agent identification
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    idx = 0
    colors = ["blue", "orange", "magenta", "cyan", "brown", "pink", "yellow", "purple"]
    group_id = {}

    def __init__(self, position, efficiency=0, color_by_efficiency=False):
        self.position = np.array(position)
        self.id = Agent.alphabet[Agent.idx]

        self.r = 0
        self.acc_r = 0
        self.r_vec = np.zeros(2)
        self.acc_r_vec = np.zeros(2)

        self.apples = 0
        self.apples_dropped = 0  # Dropped apples counter for loss inequality mode
        self.gathered = 0
        self.apples_from_box = 0
        if color_by_efficiency:
            if efficiency not in Agent.group_id:
                Agent.group_id[efficiency] = Agent.colors[len(Agent.group_id.keys())]
            self.color = Agent.group_id[efficiency]
        else:
            self.color = Agent.colors[Agent.idx]
        self.efficiency = efficiency

        Agent.idx += 1

    def reset(self, position):
        self.position = np.array(position)
        self.apples = 0
        self.apples_dropped = 0
        self.gathered = 0
        self.apples_from_box = 0
        self.r = 0
        self.r_vec = np.zeros(2)
        self.acc_r = 0
        self.acc_r_vec = np.zeros(2)

    def __gt__(self, other):
        return self.efficiency > other.efficiency

    def __lt__(self, other):
        return self.efficiency < other.efficiency

    def __eq__(self, other):
        return self.id == other.id


class MAEGG(ParallelEnv):
    # ACTION SPACE
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    STAY = 4
    DONATE = 5
    TAKE_DONATION = 6

    MOVE_VECTORS = {MOVE_UP: (-1, 0), MOVE_DOWN: (1, 0), MOVE_LEFT: (0, -1), MOVE_RIGHT: (0, 1), STAY: (0, 0),
                    DONATE: (0, 0), TAKE_DONATION: (0, 0)}

    log_level = logging.DEBUG
    metadata = {
        "name": "MultiAgentEthicalGathering-v1",
        "render_modes": ["human", "partial_observability"],
        "render_fps": 2
    }

    @staticmethod
    def read_params_from_json(json_path):
        """
        Reads the parameters from a json file and returns a dictionary with the parameters. It also checks that the
        parameters do exist in the class attributes.
        :param json_path:
        :return:
        """
        data = json.load(open(json_path))
        params = {}
        for k, v in data.items():
            if hasattr(MAEGG, k):
                params[k] = v
            else:
                raise ValueError("Parameter {} not found in class attributes".format(k))
        return params

    def __init__(self, n_agents, map_size, we: np.ndarray, inequality_mode, max_steps, donation_capacity,
                 survival_threshold, visual_radius, partial_observability, init_state="empty", track_history=False,
                 efficiency: np.ndarray = None, color_by_efficiency=False, reward_mode="scalarised", obs_mode="nn",
                 objective_order="ethical_first"):
        super(MAEGG, self).__init__()
        Agent.idx = 0
        Agent.group_id = {}

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(MAEGG.log_level)
        if not self.logger.hasHandlers():
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # Parameters
        self.n_agents = n_agents
        self.map_size = map_size
        self.we = we
        self.inequality_mode = inequality_mode
        self.max_steps = max_steps
        self.donation_capacity = donation_capacity
        self.survival_threshold = survival_threshold
        self.visual_radius = visual_radius
        self.partial_observability = partial_observability
        if self.partial_observability:
            self.render_mode = "partial_observability"
        self.color_by_efficiency = color_by_efficiency
        self.init_state = init_state
        self.reward_mode = reward_mode
        self.obejctive_order = objective_order
        self.obs_mode = obs_mode

        if self.obejctive_order not in ["individual_first", "ethical_first"]:
            raise ValueError("Objective order not recognised. Choose between 'individual_first' and 'ethical_first'")
        if self.obejctive_order == "ethical_first":
            self.we = [self.we[1], self.we[0]]

        # Variables
        self.map = Maps(sketch=self.map_size, init_state=init_state)
        self.dims = self.map.current_state.shape
        self.efficiency = efficiency
        if efficiency is None:
            self.efficiency = [i for i in range(1, self.n_agents + 1)]

        self.agents = {k: Agent(self.map.get_spawn_coords(), self.efficiency[k], color_by_efficiency) for k in
                       range(self.n_agents)}
        self.donation_box = 0
        self.steps = 0
        self.sim_data = {
            "donation_box_full": -1,
            "all_survived": -1,
            "all_done": -1,
            "time_to_survival": [-1] * self.n_agents,
            "generated_apples": 0,
        }

        # Track history
        self.track = track_history
        self.history = []
        self.stash_runs = True
        self.stash = []

        # Env Setup
        self.action_space = gym.spaces.tuple.Tuple([gym.spaces.Discrete(7)] * self.n_agents)

        if self.reward_mode == "vectorial":
            self.reward_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        elif self.reward_mode == "scalarised":
            self.reward_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        if self.partial_observability:
            self.observation_space = gym.spaces.tuple.Tuple([gym.spaces.Box(low=0, high=1,
                                                    shape=((self.visual_radius * 2 + 1) ** 2 + 2,),
                                                    dtype=np.float32)] * self.n_agents)

        else:
            self.observation_space = gym.spaces.tuple.Tuple([gym.spaces.Box(low=0, high=1, shape=(np.prod(self.map.current_state.shape) + 2,),
                                                    dtype=np.float32)] * self.n_agents)

        # Log relevant info
        self.logger.debug("Environment initialized with parameters:")
        self.logger.debug("n_agents: {}".format(self.n_agents))
        self.logger.debug("map_size: {}".format(self.map_size))
        self.logger.debug("inequality_mode: {}".format(self.inequality_mode))
        self.logger.debug("donation_capacity: {}".format(self.donation_capacity))
        self.logger.debug("survival_threshold: {}".format(self.survival_threshold))
        self.logger.debug("Weights: {}".format(self.we))

    def getObservation(self):
        """
        Returns the observation for each agent. If partial observability is enabled, the observation is a subgrid of the
        map centered on the agent and padded with wall. Otherwise, the observation is the full map. Both observations
        are normalized to be in the range [0, 1].

        Normalization values are:
        Wall -> 0
        Empty -> 0.25
        Apple -> 0.5
        Agent Self -> 0.75
        Agent Other -> 1

        :return: Observations for each agent
        """
        observations = []
        global_state = self.map.current_state.copy()

        for ag in self.agents.values():
            global_state[ag.position[0],ag.position[1]] = ag.id

        donation_box_states = list(range(self.n_agents + 1)) + [self.donation_capacity]
        normalized_db_state = bisect.bisect_right(donation_box_states, self.donation_box) - 1
        normalized_db_state /= (len(donation_box_states) - 1)

        if self.partial_observability:

            padded_state = np.pad(global_state, ((self.visual_radius, self.visual_radius),
                                                 (self.visual_radius, self.visual_radius)),
                                  mode='constant', constant_values="=")

            for ag in self.agents.values():
                x, y = ag.position
                x += self.visual_radius
                y += self.visual_radius
                obs = padded_state[x - self.visual_radius:x + self.visual_radius + 1,
                      y - self.visual_radius:y + self.visual_radius + 1]
                obs_shape = np.prod(obs.shape)
                obs = np.reshape(obs, (obs_shape,))
                normalized_obs = np.ones(obs_shape, dtype=np.float32)
                normalized_obs[obs == '@'] = 0.5
                normalized_obs[obs == ag.id] = 0.75
                normalized_obs[obs == ' '] = 0.25
                normalized_obs[obs == '='] = 0
                # Normalize donation box and survival status
                aux = np.zeros(2)
                aux[0] = normalized_db_state
                if ag.apples == 0:
                    aux[1] = 0
                elif ag.apples < self.survival_threshold:
                    aux[1] = 0.25
                elif ag.apples == self.survival_threshold:
                    aux[1] = 0.75
                else:
                    aux[1] = 1

                normalized_obs = np.concatenate((normalized_obs, aux))
                observations.append(normalized_obs)

        else:
            obs_shape = np.prod(global_state.shape)
            obs = np.reshape(global_state, (obs_shape,))
            for ag in self.agents.values():
                normalized_obs = np.ones(obs.shape, dtype=np.float32)
                normalized_obs[obs == '@'] = 0.5
                normalized_obs[obs == ' '] = 0.25
                normalized_obs[obs == ag.id] = 0.75
                aux = np.zeros(2)
                aux[0] = normalized_db_state
                if ag.apples == 0:
                    aux[1] = 0
                elif ag.apples < self.survival_threshold:
                    aux[1] = 0.25
                elif ag.apples == self.survival_threshold:
                    aux[1] = 0.75
                else:
                    aux[1] = 1

                normalized_obs = np.concatenate((normalized_obs, aux))
                observations.append(normalized_obs.astype(np.float32))

        if self.obs_mode == "cnn":
            obs = []
            for o in observations:
                # Get map sketch size
                map_dims = self.map.current_state.shape
                po_dims = (self.visual_radius * 2 + 1, self.visual_radius * 2 + 1)
                obs.append({
                    "image": o[:-2].reshape(po_dims) if self.partial_observability else o[:-2].reshape(map_dims),
                    "donation_box": o[-2],
                    "survival_status": o[-1]
                })
            return obs
        for obs in observations:
            obs = obs.astype(np.float32)
        return tuple(observations)

    def step(self, action):
        """
        Takes a step in the environment. The action is a list of actions, one for each agent. The actions are executed.
        Things to take into account:
        - Agents can't move to a wall
        - Agent can't donate more than the donation capacity
        - Agents can't donate if they don't have enough apples
        - Agents can't take donations if box is empty
        - Agents action order specified by self.inequality_mode
        - Regen apples
        :param action:
        :return: nObservations, reward, done, info
        """
        info = {
            "R'_N": [0] * self.n_agents,
            "R'_E": [0] * self.n_agents,
        }

        self.steps += 1
        done = self.steps >= self.max_steps

        reward = np.zeros((self.n_agents, 2))

        untie_prio = np.random.permutation(self.n_agents)
        sorted_pairs = sorted(zip(self.agents.values(), action, untie_prio, list(range(self.n_agents))), reverse=True,
                              key=lambda x: (x[0].efficiency, x[2]))
        sorted_agents, action, _, idx = map(list, zip(*sorted_pairs))
        for i in range(self.n_agents):
            events = self.get_action_events(sorted_agents[i], action[i])
            info[sorted_agents[i].id] = {"events": events, "apples": sorted_agents[i].apples}
            if 'did_not_donate' in events:
                info["R'_E"][idx[i]] += 1
            if 'donated' in events:
                reward[i, 1] += 0.7
                reward[i, 0] -= 1
            if 'suboptimal' in events:
                reward[i, 0] -= 1

            if 'greedy' in events:
                reward[i, 1] += -1.0
                info["R'_N"][idx[i]] += 1
            if 'hungry' in events:
                reward[i, 0] += -1.0
            if 'picked_apple' in events:
                reward[i, 0] += 1.0

            if self.obejctive_order == "ethical_first":
                reward = np.vstack([reward[:, 1], reward[:, 0]]).T

            if self.reward_mode == "scalarised":
                sorted_agents[i].r = np.dot(reward[i], self.we)
            elif self.reward_mode == "vectorial":
                sorted_agents[i].r = reward[i]
            else:
                raise ValueError("Reward mode not recognised")
            sorted_agents[i].r_vec = reward[i]
            sorted_agents[i].acc_r += sorted_agents[i].r
            sorted_agents[i].acc_r_vec += reward[i]

        self.sim_data["generated_apples"] += self.map.regen_apples(self.agents.values())
        nObservations = self.getObservation()

        if self.sim_data["donation_box_full"] == -1 and self.donation_box == self.donation_capacity:
            self.sim_data["donation_box_full"] = self.steps

        # Time to survival
        for i, ag in enumerate(self.agents.values()):
            if self.sim_data["time_to_survival"][i] == -1 and ag.apples >= self.survival_threshold:
                self.sim_data["time_to_survival"][i] = self.steps

        if self.sim_data["all_survived"] == -1 and all(
                [ag.apples >= self.survival_threshold for ag in self.agents.values()]):
            self.sim_data["all_survived"] = self.steps

        if self.sim_data["all_done"] == -1 and self.sim_data["donation_box_full"] != -1 and self.sim_data[
            "all_survived"] != -1:
            self.sim_data["all_done"] = self.steps

        info["sim_data"] = self.sim_data
        info["donation_box"] = self.donation_box

        if self.track:
            self.history.append(info)

        return nObservations, np.array([float(ag.r) for ag in self.agents.values()]), done, False, info

    def reset(self, seed=None, options=None):
        """
        Resets the environment. The map is reset, the agents are placed in the map and the donation box is emptied.
        :return:
        """
        if seed is not None:
            np.random.seed(seed)

        self.map.reset()

        for ag in self.agents.values():
            ag.reset(self.map.get_spawn_coords())

        self.sim_data = {
            "donation_box_full": -1,
            "all_survived": -1,
            "all_done": -1,
            "time_to_survival": [-1] * self.n_agents,
            "generated_apples": 0,
        }

        self.donation_box = 0
        self.steps = 0

        if self.track and self.stash_runs and len(self.history) > 0:
            self.stash.append(self.build_history_array())
        self.history = []
        self.gen_apples = 0
        return self.getObservation(), {}

    def render(self, mode="human", pause=0.03):
        frame = self.map.current_state.copy()

        if mode == "text":
            for ag in self.agents.values():
                frame[ag.position[0],ag.position[1]] = ag.id
            print(frame)

        elif mode == "human":
            rgb_frame = np.zeros((frame.shape[0], frame.shape[1], 3))
            rgb_frame[frame == '@'] = [0, 1, 0]
            for ag in self.agents.values():
                rgb_frame[ag.position[0], ag.position[1], :] = [1, 0, 0]
            plt.figure(pause)
            plt.ion()
            plt.imshow(rgb_frame)
            plt.axis('off')
            # Set number of apples per agent and donation box as title
            text = ""
            for ag in self.agents.values():
                text += "{}: {} ".format(ag.id, ag.apples)
            text += "Donation Box: {}".format(self.donation_box)
            plt.title(text)
            plt.pause(pause)
            plt.clf()

        elif mode == "partial_observability":
            if not self.partial_observability:
                raise ValueError("Partial observability is disabled for full observability maps")
            rgb_frame = np.zeros((frame.shape[0], frame.shape[1], 3))
            rgb_frame[frame == '@'] = [0, 1, 0]
            for ag in self.agents.values():
                rgb_frame[ag.position[0], ag.position[1], :] = [1, 0, 0]

            # Set to grey the cells that are not visible
            mask = np.zeros((*rgb_frame.shape[:-1], self.n_agents))
            for k, ag in enumerate(self.agents.values()):
                x, y = ag.position
                xo = max(x - self.visual_radius, 0)
                yo = max(y - self.visual_radius, 0)
                x1 = min(x + self.visual_radius + 1, rgb_frame.shape[0])
                y1 = min(y + self.visual_radius + 1, rgb_frame.shape[1])
                mask[xo:x1, yo:y1, k] = 1

            mask = np.any(mask, axis=2)

            # Set number of apples per agent and donation box as title
            text = ""
            for ag in self.agents.values():
                text += "{}: {} ".format(ag.id, ag.apples)
            text += "Donation Box: {}".format(self.donation_box)

            plt.figure(1)
            plt.subplot(1, 2, 1)
            plt.imshow(rgb_frame)
            plt.axis("off")
            plt.title(text)
            plt.subplot(1, 2, 2)
            rgb_frame[mask == 0] = [0.5, 0.5, 0.5]
            plt.imshow(rgb_frame)
            plt.axis("off")
            plt.title("Masked")
            plt.show(block=False)
            plt.pause(0.05)
            plt.clf()
        pass

    def get_action_events(self, agent, action):
        """
        Uses map status and action to update agent state and return the events that happened as a result of the action.
        The events include:
        - Agent moved
        - Agent picked apple
        - Agent donated
        - Agent took donation
        - Agent above survival threshold
        - Agent did unethical action
        :param agent:
        :param action:
        :return: Set of events that happened as a result of the action.
        """
        events = set()
        move_vec = MAEGG.MOVE_VECTORS[action]
        # Could have donated
        if action != MAEGG.DONATE and agent.apples > self.survival_threshold and self.donation_box < self.donation_capacity:
            events.add("did_not_donate")

        if self.map.check_valid_position(agent.position + move_vec):
            agent.position += move_vec
            events.add("moved")
        # Did agent step on apple?
        if self.map.current_state[agent.position[0],agent.position[1]] == '@':
            self.map.current_state[agent.position[0],agent.position[1]] = ' '
            if self.inequality_mode == "loss":
                if np.random.rand() < agent.efficiency:
                    agent.apples += 1
                    agent.gathered += 1
                    events.add("picked_apple")
                else:
                    agent.apples_dropped += 1
            else:
                agent.apples += 1
                agent.gathered += 1
                events.add("picked_apple")

        if action == MAEGG.TAKE_DONATION:
            if agent.apples >= self.survival_threshold:
                events.add("greedy")
            if self.donation_box > 0:
                agent.apples += 1
                agent.apples_from_box += 1
                self.donation_box -= 1
                events.add("took_donation")

        elif action == MAEGG.DONATE:
            if agent.apples > 0 and self.donation_box < self.donation_capacity:
                self.donation_box += 1
                agent.apples -= 1
                if agent.apples < self.survival_threshold:
                    events.add("suboptimal")
                else:
                    events.add("donated")

        if agent.apples >= self.survival_threshold:
            events.add("not hungry")
        else:
            events.add("hungry")
        return events

    def build_history_array(self, h=None):
        """
        Builds an array of shape (n_steps, donation_box, apples_agent0, ..., apples_agentN) with the history of the environment.
        :return:
        """
        from collections import Counter
        from itertools import chain

        if len(self.history) == 0 and h is None:
            raise ValueError("No history to retrieve")
        history_array = np.zeros((self.max_steps, self.n_agents + 1))
        event_histogram = [[] for _ in range(self.n_agents)]

        for i, step in enumerate(self.history if h is None else h):
            history_array[i, 0] = step['donation_box']
            for j, ag in enumerate(self.agents.values()):
                history_array[i, j + 1] = step[ag.id]['apples']
                event_histogram[j].append(step[ag.id]['events'])
        for i, l in enumerate(event_histogram):
            event_histogram[i] = Counter(chain.from_iterable(event_histogram[i]))

        return history_array, event_histogram

    def plot_results(self, type="histogram", save_path=None, show=True):
        """
        Plots the results of the environment. It plots the history of the donation box and the number of apples for each
        agent in each step, or a histogram with the events that happened to each agent.
        Plots:
        - Histogram: histogram of events that happened to each agent. Works with self.history (last run)
        - Line: line plot of donation box and number of apples for each agent in each step. Works with self.history
        - Median: line plot of median+IQR number of apples for each agent in each step. Works with self.stash
        - Mean: line plot of mean+std number of apples for each agent in each step. Works with self.stash
        - Blurr: line plot of number of apples for each agent in each step with low alpha. Works with self.stash
        :return:
        """
        colors = [ag.color for ag in self.agents.values()]
        if type == "histogram":
            history_array, event_histogram = self.build_history_array()
            # Remove irrelevant events from the histogram and register all tags
            all_tags = set()
            for h in event_histogram:
                if 'not hungry' in h:
                    h.pop('not hungry')
                h.pop('moved')
                all_tags = all_tags.union(set(h.keys()))
            for h in range(len(event_histogram)):
                event_histogram[h] = {tag: event_histogram[h].get(tag, 0) for tag in all_tags}

            # Set up the positions for the bars
            x_positions = range(len(all_tags))
            width = 0.8 / self.n_agents

            # Make the histogram
            fig, ax = plt.subplots()
            for i in range(0, len(event_histogram)):
                tags, counts = zip(*sorted(event_histogram[i].items()))
                bars = ax.bar([x + width * i for x in x_positions], counts, width, label='Agent {}'.format(i),
                              color=colors[i])
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width() / 2, height, str(height),
                             ha='center', va='bottom', color='k')

            # Optionally, add labels, a title, and a legend
            plt.xlabel('Events')
            plt.ylabel('Counts')
            plt.title('Event Histogram')
            plt.xticks(ticks=x_positions, labels=tags)  # Align x-axis labels with the center of the grouped bars
            plt.legend()

        if type == "line":
            if len(self.history) == 0:
                history_array, event_histogram = self.stash[-1]
            else:
                history_array, event_histogram = self.build_history_array()
            for i in range(1, self.n_agents + 1):
                plt.plot(history_array[:, i], label="Agent {}".format(i), color=colors[i - 1])
            plt.plot(history_array[:, 0], label="Donation Box", color='green')
            # Plot survival threshold and donation capacity
            plt.plot([0, self.max_steps], [self.survival_threshold, self.survival_threshold],
                     label="Survival Threshold", linestyle='--', color='red')
            plt.plot([0, self.max_steps], [self.donation_capacity, self.donation_capacity], label="Donation Capacity",
                     linestyle='--', color='black')
            plt.legend()

        if type == "median":
            apple_history = self.get_results("apple_history")
            median = np.median(apple_history, axis=0)
            iqr = np.percentile(apple_history, 75, axis=0) - np.percentile(apple_history, 25, axis=0)
            seen_groups = set()
            # Set y axis limits
            plt.ylim(0, 65)
            for i in range(1, self.n_agents + 1):
                label = "Agent {}".format(i)
                if self.color_by_efficiency:
                    level = "High" if self.efficiency[i - 1] > 0.5 else "Low"
                    label = "{} Efficiency Group".format(level)
                    if label in seen_groups:
                        label = "_nolegend_"  # Don't add the same label twice
                    else:
                        seen_groups.add(label)
                plt.plot(median[:, i], label=label, color=colors[i - 1])
                plt.fill_between(range(self.max_steps), median[:, i] - iqr[:, i], median[:, i] + iqr[:, i],
                                 alpha=0.2, color=colors[i - 1])
            plt.plot(median[:, 0], label="Donation Box", color='green')
            plt.fill_between(range(self.max_steps), median[:, 0] - iqr[:, 0], median[:, 0] + iqr[:, 0],
                             alpha=0.2, color='green')
            # Plot survival threshold and donation capacity
            plt.plot([0, self.max_steps], [self.survival_threshold, self.survival_threshold],
                     label="Survival Threshold", linestyle='--', color='red')
            plt.plot([0, self.max_steps], [self.donation_capacity, self.donation_capacity], label="Donation Capacity",
                     linestyle='--', color='black')
            plt.legend()

        if type == "mean":
            apple_history = self.get_results("apple_history")
            mean = np.mean(apple_history, axis=0)
            std = np.std(apple_history, axis=0)
            for i in range(1, self.n_agents + 1):
                plt.plot(mean[:, i], label="Agent {}".format(i), color=colors[i - 1])
                plt.fill_between(range(self.max_steps), mean[:, i] - std[:, i], mean[:, i] + std[:, i],
                                 alpha=0.2, color=colors[i - 1])
            plt.plot(mean[:, 0], label="Donation Box", color='green')
            plt.fill_between(range(self.max_steps), mean[:, 0] - std[:, 0], mean[:, 0] + std[:, 0],
                             alpha=0.2, color='green')
            # Plot survival threshold and donation capacity
            plt.plot([0, self.max_steps], [self.survival_threshold, self.survival_threshold],
                     label="Survival Threshold", linestyle='--', color='red')
            plt.plot([0, self.max_steps], [self.donation_capacity, self.donation_capacity], label="Donation Capacity",
                     linestyle='--', color='black')
            plt.legend()

        if type == "blurr":

            if len(self.history) > 0:
                self.stash.append(self.build_history_array())
            self.history = []

            # Plot blurr
            raise NotImplementedError("Blurr plot not implemented yet")

        if save_path is not None:
            plt.savefig(save_path)

        if show:
            plt.show()
        plt.clf()

    def print_results(self):
        """
        :return:
        """
        results = {}
        header, histogram = self.get_results("histogram")
        results["histogram_data"] = histogram
        results["histogram_header"] = header

        # Pretty print histogram for each agent in different columns
        table = prettytable.PrettyTable()
        table.field_names = ["Agent"] + header

        for i in range(self.n_agents):
            table.add_row([i] + [c for c in histogram[i]])

        title = "Average Event Histogram"
        print("=" * len(title))
        print(title)
        print("=" * len(title))

        print(table)
        return results

    def get_results(self, type="histogram"):
        if type == "histogram":
            if len(self.history) > 0:
                self.stash.append(self.build_history_array())
            self.history = []

            all_tags = set()

            # Combine histogram from stash
            event_histogram = []
            for h in self.stash:
                event_histogram.append(h[1])
                for c in h[1]:
                    all_tags = all_tags.union(set(c.keys()))

            # Remove irrelevant events from the histogram and register all tags
            all_tags.discard('not hungry')
            all_tags.discard('moved')

            all_tags = sorted(list(all_tags))

            counter = np.zeros((self.n_agents, len(all_tags)), dtype=np.float32)
            for h in event_histogram:
                for i in range(self.n_agents):
                    counter[i] += np.array([h[i][tag] for tag in all_tags])
            counter = counter / len(self.stash)
            counter = np.round(counter, 2)

            return all_tags, counter

        if type == "apple_history":
            if len(self.history) == 0 and len(self.stash) == 0:
                raise ValueError("No history to retrieve")
            if len(self.history) > 0:
                self.stash.append(self.build_history_array())
            self.history = []

            apple_history = np.array([s[0] for s in self.stash])
            if apple_history.shape[0] == 0:
                raise ValueError("No history to retrieve")
            return apple_history

    def toggleTrack(self, track):
        self.track = track

    def toggleStash(self, stash_runs):
        self.stash_runs = stash_runs

    def setStash(self, stash):
        self.stash = stash

    def setHistory(self, history):
        if type(history) == list:
            self.history = history
        else:
            self.history = [history]

    def resetStash(self):
        self.stash = []
        self.history = []