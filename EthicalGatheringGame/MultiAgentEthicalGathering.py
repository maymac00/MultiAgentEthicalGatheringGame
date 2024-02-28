from collections import deque

import numpy as np

from matplotlib import pyplot as plt

import gym
import json
from EthicalGatheringGame.Maps import Maps
import bisect
import logging


# TODO: Check env.Wrapper subclassing to achieve: action space mapping, callbacks, last action memory, etc. This will keep the base env simpler
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
        self.r_vec = np.zeros(2)

        self.apples = 0
        self.gather_tries = 0
        self.gathered = 0
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
        self.gather_tries = 0
        self.gathered = 0
        self.r = 0
        self.r_vec = np.zeros(2)

    def __gt__(self, other):
        return self.efficiency > other.efficiency

    def __lt__(self, other):
        return self.efficiency < other.efficiency

    def __eq__(self, other):
        return self.id == other.id


class MAEGG(gym.Env):
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
                 efficiency: np.ndarray = None, color_by_efficiency=False):
        super(MAEGG, self).__init__()
        Agent.idx = 0
        Agent.group_id = {}

        # Logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)
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
        self.color_by_efficiency = color_by_efficiency
        self.init_state = init_state

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
        }

        # Track history
        self.track = track_history
        self.history = []
        self.stash_runs = True
        self.stash = []
        self.gen_apples = 0  # Generated apples
        self.apple_gen_statistic = deque(maxlen=100)

        # Env Setup
        self.action_space = gym.spaces.Discrete(7)
        if self.partial_observability:
            self.observation_space = gym.spaces.Box(low=0, high=1,
                                                    shape=((self.visual_radius * 2 + 1) ** 2 + 2,),
                                                    dtype=np.float32)

        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(np.prod(self.map.current_state.shape) + 2,),
                                                    dtype=np.float32)

        # Log relevant info
        self.logger.info("Environment initialized with parameters:")
        self.logger.info("n_agents: {}".format(self.n_agents))
        self.logger.info("map_size: {}".format(self.map_size))
        self.logger.info("inequality_mode: {}".format(self.inequality_mode))
        self.logger.info("donation_capacity: {}".format(self.donation_capacity))
        self.logger.info("survival_threshold: {}".format(self.survival_threshold))
        self.logger.info("Weights: {}".format(self.we))

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
            global_state[*ag.position] = ag.id

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
                normalized_obs = np.ones(obs_shape)
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
            return observations

        else:
            obs_shape = np.prod(global_state.shape)
            obs = np.reshape(global_state, (obs_shape,))
            for ag in self.agents.values():
                normalized_obs = np.ones(obs.shape)
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
                observations.append(normalized_obs)

            return observations

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
        info = {}

        self.steps += 1
        done = self.steps >= self.max_steps

        reward = np.zeros((self.n_agents, 2))

        untie_prio = np.random.permutation(self.n_agents)
        sorted_pairs = sorted(zip(self.agents.values(), action, untie_prio), reverse=True,
                              key=lambda x: (x[0].efficiency, x[2]))
        sorted_agents, action, _ = map(list, zip(*sorted_pairs))
        for i in range(self.n_agents):
            events = self.get_action_events(sorted_agents[i], action[i])
            info[sorted_agents[i].id] = {"events": events, "apples": sorted_agents[i].apples}
            if 'donated' in events:
                reward[i, 1] += 0.7
                reward[i, 0] -= 1
            if 'suboptimal' in events:
                reward[i, 0] -= 1

            if 'greedy' in events:
                reward[i, 1] += -1.0
            if 'hungry' in events:
                reward[i, 0] += -1.0
            if 'picked_apple' in events:
                reward[i, 0] += 1.0

            sorted_agents[i].r = np.dot(reward[i], self.we)
            sorted_agents[i].r_vec += reward[i]

        generated_apples = self.map.regen_apples(self.agents.values())
        self.gen_apples += generated_apples
        nObservations = self.getObservation()

        if self.sim_data["donation_box_full"] == -1 and self.donation_box == self.donation_capacity:
            self.sim_data["donation_box_full"] = self.steps

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

        return nObservations, np.array([ag.r for ag in self.agents.values()]), [done] * self.n_agents, info

    def reset(self, seed=None):
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
        }

        self.donation_box = 0
        self.steps = 0

        if self.track and self.stash_runs and len(self.history) > 0:
            self.stash.append(self.build_history_array())
        self.history = []
        self.apple_gen_statistic.append(self.gen_apples)
        self.gen_apples = 0
        return (self.getObservation(), {
            "donation_box": self.donation_box,
            "steps": self.steps,
        })

    def render(self, mode="human", pause=0.03):
        frame = self.map.current_state.copy()

        if mode == "text":
            for ag in self.agents.values():
                frame[*ag.position] = ag.id
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
        if self.map.check_valid_position(agent.position + move_vec):
            agent.position += move_vec
            events.add("moved")
        # Did agent step on apple?
        if self.map.current_state[*agent.position] == '@':
            self.map.current_state[*agent.position] = ' '
            agent.gather_tries += 1
            if self.inequality_mode == "loss":
                if np.random.binomial(agent.efficiency, 1 / self.n_agents) > 0:
                    agent.apples += 1
                    agent.gathered += 1
                    events.add("picked_apple")
            else:
                agent.apples += 1
                agent.gathered += 1
                events.add("picked_apple")

        if action == MAEGG.TAKE_DONATION:
            if agent.apples >= self.survival_threshold:
                events.add("greedy")
            if self.donation_box > 0:
                agent.apples += 1
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

    def plot_results(self, type="histogram"):
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
            history_array, event_histogram = self.build_history_array()
            for i in range(1, self.n_agents + 1):
                plt.plot(history_array[:, i], label="Agent {}".format(i - 1), color=colors[i - 1])
            plt.plot(history_array[:, 0], label="Donation Box", color='green')
            # Plot survival threshold and donation capacity
            plt.plot([0, self.max_steps], [self.survival_threshold, self.survival_threshold],
                     label="Survival Threshold", linestyle='--', color='red')
            plt.plot([0, self.max_steps], [self.donation_capacity, self.donation_capacity], label="Donation Capacity",
                     linestyle='--', color='black')
            plt.legend()

        if type == "median":

            if len(self.history) == 0 and len(self.stash) == 0:
                raise ValueError("No history to plot")
            if len(self.history) > 0:
                self.stash.append(self.build_history_array())
            self.history = []

            # Plot median + IQR
            apple_history = np.array([s[0] for s in self.stash])
            median = np.median(apple_history, axis=0)
            iqr = np.percentile(apple_history, 75, axis=0) - np.percentile(apple_history, 25, axis=0)
            seen_groups = set()
            for i in range(1, self.n_agents + 1):
                label = "Agent {}".format(i - 1)
                if self.color_by_efficiency:
                    label = "Group Efficiency {}".format(self.efficiency[i - 1])
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

            if len(self.history) > 0:
                self.stash.append(self.build_history_array())
            self.history = []

            # Plot mean + std
            apple_history = np.array([s[0] for s in self.stash])
            mean = np.mean(apple_history, axis=0)
            std = np.std(apple_history, axis=0)
            for i in range(1, self.n_agents + 1):
                plt.plot(mean[:, i], label="Agent {}".format(i - 1), color=colors[i - 1])
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
        plt.show()

    def print_results(self):
        """

        :return:
        """
        header, histogram = self.get_results("histogram")
        # Pretty print histogram for each agent in different columns
        print(f"Agent | {' | '.join([tag.ljust(13) for tag in header])}")
        print("-" * (15 + 15 * len(histogram[0])))
        for i in range(self.n_agents):
            print(f"{i}     | {' | '.join([str(c).ljust(13) for c in histogram[i]])}")

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

            counter = np.zeros((self.n_agents, len(all_tags)), dtype=np.float32)
            for h in event_histogram:
                for i in range(self.n_agents):
                    counter[i] += np.array([h[i][tag] for tag in all_tags])
            counter = counter / len(self.stash)
            counter = np.round(counter, 2)

            return all_tags, counter

    def setTrack(self, track):
        self.track = track

    def setStash(self, stash_runs):
        self.stash_runs = stash_runs

    def resetStash(self):
        self.stash = []
        self.history = []
