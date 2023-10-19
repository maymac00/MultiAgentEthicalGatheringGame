import numpy as np


class Maps:
    sketches = {
        "tiny": np.array([
            list(' @@ '),
            list(' @  '),
            list('    ')]),
        "small": np.array([
            list('     '),
            list('  @@ '),
            list('  @  '),
            list('     ')]),
        "large": np.array([
            list('                                      '),
            list('             @      @@@@@       @     '),
            list('         @   @@         @@@    @  @   '),
            list('      @ @@@  @@@    @    @ @@ @@@@    '),
            list('  @  @@@ @    @  @ @@@  @  @   @ @    '),
            list(' @@@  @ @    @  @@@ @  @@@        @   '),
            list('  @ @  @@@  @@@  @ @    @ @@   @@ @@  '),
            list('   @ @  @@@    @ @  @@@    @@@  @     '),
            list('    @@@  @      @@@  @    @@@@        '),
            list('     @       @  @ @@@    @  @         '),
            list(' @  @@@  @  @  @@@ @    @@@@          '),
            list('     @ @   @@@  @ @      @ @@   @     '),
            list('      @@@   @ @  @@@      @@   @@@    '),
            list('  @    @     @@@  @             @     '),
            list('              @                       '),
            list('                                      ')
        ]),
    }
    sketches["medium"] = sketches["large"][:, :16]
    apple_regen = {
        "tiny": 0.05,
        "small": 0.05,
        "medium": 0.005,
        "large": 0.005
    }

    def __init__(self, sketch, init_state="empty"):
        self.sketch = sketch
        self.apple_regen = Maps.apple_regen[sketch]
        self.init_state = init_state

        self.current_state = Maps.sketches[self.sketch]
        self.apple_cells = np.array(np.where(self.current_state == '@')).T
        self.spawn_able_cells = np.where(self.current_state == ' ')

        if self.init_state == "empty":
            self.current_state[self.current_state == '@'] = ' '
        elif self.init_state == "random":
            self.current_state[self.current_state == '@'] = np.random.choice([' ', '@'], size=self.current_state.shape,
                                                                             p=[1 - self.apple_regen, self.apple_regen])
        elif self.init_state == "full":
            pass
        else:
            raise ValueError("init_state must be one of 'empty', 'random', 'full'")

    def reset(self):
        if self.init_state == "empty":
            self.current_state[self.current_state == '@'] = ' '
        elif self.init_state == "random":
            self.current_state[self.current_state == '@'] = np.random.choice([' ', '@'], size=self.current_state.shape,
                                                                             p=[1 - self.apple_regen, self.apple_regen])
        elif self.init_state == "full":
            pass
        else:
            raise ValueError("init_state must be one of 'empty', 'random', 'full'")

    def get_spawn_coords(self):
        return np.random.choice(self.spawn_able_cells[0]), np.random.choice(self.spawn_able_cells[1])

    def check_valid_position(self, pos):
        if pos[0] < 0 or pos[0] >= self.current_state.shape[0]:
            return False
        if pos[1] < 0 or pos[1] >= self.current_state.shape[1]:
            return False
        return True

    def __getitem__(self, tup):
        return self.current_state[tup]

    def regen_apples(self, agents):
        """
        Regenerates apples on the map with probability apple_regen. Using apple cells as a mask, we generate a random binomial
        distribution with probability apple_regen and replace the apple cells with the result. Apples should not appear
        on cells that are currently occupied by an agent.
        :return:
        """
        mask = np.random.binomial(1, self.apple_regen, size=self.apple_cells.shape[0])
        # If an agent is on the cell, we don't want to spawn an apple there
        for agent in agents:
            aux = self.apple_cells != agent.position
            mask = np.logical_and(mask, aux.sum(axis=1) > 0)

        # If an apple is already on the cell, we want to keep it there
        final_value = np.logical_or(mask, self.current_state[*self.apple_cells.T] == '@')
        self.current_state[*self.apple_cells.T] = np.where(final_value, '@', ' ')
