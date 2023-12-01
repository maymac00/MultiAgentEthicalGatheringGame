import unittest
from MultiAgentEthicalGathering import MAEGG
from presets import tiny, small, medium, large
import numpy as np


class TestMAEGG(unittest.TestCase):
    def setUp(self):
        self.env = MAEGG(**tiny)
        self.env.reset()  # Reset environment to a starting state

    def test_action_apples(self):
        # Assuming you can set the state directly
        # Set the environment to a specific state (example state)
        def reset():
            self.env.map.spawn_all_apples()
            self.env.map.current_state[0, 1] = ' '

            # set agent positions
            self.env.agents[0].position = np.array([0, 1])
            self.env.agents[1].position = np.array([1, 2])

            # set agent apples to 0
            self.env.agents[0].apples = 0
            self.env.agents[1].apples = 0

        reset()

        global_state = self.env.map.current_state.copy()
        for ag in (self.env.agents.values()):
            global_state[*ag.position] = ag.id

        actions = [[1, 0], [4, 4], [3, 0], [1, 2], [2, 3], [1, 0], [0, 4], [4, 1]]
        target_apples = [[1 ,1], [0, 0], [0, 1], [0, 1], [0, 0], [1, 1], [0, 0], [0, 0]]
        for a, t in zip(actions, target_apples):
            next_state, reward, done, _ = self.env.step(a)  # Perform action
            # Check agent apples
            self.assertEqual(self.env.agents[0].apples, t[0])
            self.assertEqual(self.env.agents[1].apples, t[1])

            reset()


if __name__ == '__main__':
    unittest.main()
