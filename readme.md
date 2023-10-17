## Multi-Agent Ethical Gathering Game

### Overview
This code is for a custom Gym environment called `MultiAgentEthicalGathering`. The environment consists of a 2D grid where multiple agents can move and pick apples. Agents can also donate apples to a common "donation box" and take apples from it.

### Requirements
- Python 3.x
- NumPy
- Gym
- Matplotlib

### Classes

#### Agent
- `__init__(self, position, efficiency=0)`: Initialize the agent with position and efficiency.
- `reset(self, position)`: Reset the agent's state.
- `__gt__(self, other)`, `__lt__(self, other)`, `__eq__(self, other)`: Comparison functions based on efficiency.

#### MAEGG (Environment)
- `__init__(self, n_agents, map_size, ...)`: Initialize the environment with various parameters.
- `getObservation()`: Return the state of the environment from each agent's perspective.
- `step(self, action)`: Take a step in the environment.
- `reset(self, seed=None)`: Reset the environment.
- `render(self, mode="human", pause=0.1)`: Render the environment.
- `get_action_events(self, agent, action)`: Calculate the events that happened as a result of an action.

### Example Usage

```python
from gym.envs.registration import register

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
    env.step([np.random.randint(0, 6) for _ in range(5)])
    env.render()
