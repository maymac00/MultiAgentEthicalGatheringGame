import numpy as np

from matplotlib import pyplot as plt

import gym
import json
from EthicalGatheringGame.Maps import Maps


class Agent:
    # Alphabet for agent identification
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    idx = 0
    colors = ["blue", "orange", "magenta", "cyan", "brown", "pink", "yellow"]

    def __init__(self, position, efficiency=0):
        self.position = np.array(position)
        self.id = Agent.alphabet[Agent.idx]
        self.color = Agent.colors[Agent.idx]

        self.r = 0

        self.apples = 0
        self.efficiency = efficiency

        Agent.idx += 1

    def reset(self, position):
        self.position = np.array(position)
        self.apples = 0

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
                 efficiency: np.ndarray = None):
        super(MAEGG, self).__init__()
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
        self.init_state = init_state

        # Variables
        self.map = Maps(sketch=self.map_size, init_state=init_state)
        self.dims = self.map.current_state.shape
        self.efficiency = efficiency
        if efficiency is None:
            self.efficiency = [i for i in range(1, self.n_agents + 1)]

        self.agents = {k: Agent(self.map.get_spawn_coords(), self.efficiency[k]) for k in range(self.n_agents)}
        self.donation_box = 0
        self.steps = 0

        # Track history
        self.track = track_history
        self.history = []
        self.stash_runs = True
        self.stash = []

        # Env Setup
        self.action_space = gym.spaces.Discrete(7)
        if self.partial_observability:
            self.observation_space = gym.spaces.Box(low=0, high=1,
                                                    shape=((self.visual_radius * 2 + 1) ** 2 + 2,),
                                                    dtype=np.float32)

        else:
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(np.prod(self.map.current_state.shape) + 2,),
                                                    dtype=np.float32)

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

        donation_box_states = list(range(self.n_agents + 1)) + [self.donation_capacity - 1, self.donation_capacity]
        donation_box_states = list(dict.fromkeys(donation_box_states).keys())
        normalized_db_state = np.digitize(self.donation_box, donation_box_states) - 1
        if normalized_db_state == len(donation_box_states) - 1:
            normalized_db_state = len(donation_box_states) - 2

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
            if 'donate' in events:
                reward[i, 1] += 0.7
            if 'greedy' in events:
                reward[i, 1] += -1.0
            if 'hungry' in events:
                reward[i, 0] += -1.0
            if 'picked_apple' in events:
                reward[i, 0] += 1.0

            sorted_agents[i].r = np.dot(reward[i], self.we)

        self.map.regen_apples(self.agents.values())
        nObservations = self.getObservation()
        info["donation_box"] = self.donation_box

        if self.track:
            self.history.append(info)

        return nObservations, np.array([ag.r for ag in self.agents.values()]), done, info

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

        self.donation_box = 0
        self.steps = 0

        if self.track and self.stash_runs and len(self.history) > 0:
            self.stash.append(self.build_history_array())
        self.history = []

        return (self.getObservation(), {
            "donation_box": self.donation_box,
            "steps": self.steps,
        })

    def render(self, mode="human", pause=0.03):
        frame = self.map.current_state.copy()

        if mode == "text":
            for ag in self.agents.values():
                frame[self.agents[ag]] = ag.id
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
            if self.inequality_mode == "loss":
                if np.random.binomial(agent.efficiency, 1 / self.n_agents) > 0:
                    agent.apples += 1
                    events.add("picked_apple")
            else:
                agent.apples += 1
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

            if len(self.history) > 0:
                self.stash.append(self.build_history_array())
            self.history = []

            # Plot median + IQR
            apple_history = np.array([s[0] for s in self.stash])
            median = np.median(apple_history, axis=0)
            iqr = np.percentile(apple_history, 75, axis=0) - np.percentile(apple_history, 25, axis=0)
            for i in range(1, self.n_agents + 1):
                plt.plot(median[:, i], label="Agent {}".format(i - 1), color=colors[i - 1])
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
