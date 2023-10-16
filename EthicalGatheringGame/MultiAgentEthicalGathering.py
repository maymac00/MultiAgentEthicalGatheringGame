import numpy as np

import matplotlib

matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt

import gym

from EthicalGatheringGame.Maps import Maps


class Agent:
    # Alphabet for agent identification
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    idx = 0

    def __init__(self, position, efficiency=0):
        self.position = np.array(position)
        self.id = Agent.alphabet[Agent.idx]
        Agent.idx += 1

        self.apples = 0
        self.efficiency = efficiency

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

    def __init__(self, n_agents, map_size, we: np.ndarray, inequality_mode, max_steps, apple_regen, donation_capacity,
                 survival_threshold, visual_radius, partial_observability, init_state="empty", track_history=True):
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
        self.agents = {k: Agent(self.map.get_spawn_coords(), k) for k in range(self.n_agents)}
        # TODO: parametrize efficiency
        self.donation_box = 0
        self.steps = 0

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

        sorted_pairs = sorted(zip(self.agents.values(), action), reverse=True)
        sorted_agents, action = map(list, zip(*sorted_pairs))

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

        self.map.regen_apples()
        nObservations = self.getObservation()
        reward = np.dot(reward, self.we)
        return nObservations, reward, done, info

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
        return (self.getObservation(), {
            "donation_box": self.donation_box,
            "steps": self.steps,
        })

    def render(self, mode="human", pause=0.1):
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
            plt.title('Ethical Gathering')  # TODO: add more info
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
            # TODO: add inequality variant
            if self.map.current_state[*agent.position] == '@':
                agent.apples += 1
                self.map.current_state[*agent.position] = ' '
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

    def _check_param(self, param):
        pass

# Your custom implementation
if __name__ == "__main__":
    # Register the environment
    from gym.envs.registration import register
    register(
        id='MultiAgentEthicalGathering-v1',
        entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG')

    config_dict = {
        'n_agents': 3,
        'map_size': 'very_large',
        'we': [1, 2.6],
        'inequality_mode': 'tie',
        'max_steps': 500,
        'apple_regen': 0.05,
        'donation_capacity': 10,
        'survival_threshold': 10,
        'visual_radius': 5,
        'partial_observability': True,
        'init_state': 'full'
    }

    env = gym.make("MultiAgentEthicalGathering-v1", **config_dict)
    env.reset()
    for i in range(10):
        env.getObservation()
        env.step([np.random.randint(0, 5) for _ in range(5)])
        env.render()
    plt.show()
