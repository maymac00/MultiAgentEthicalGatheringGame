import numpy as np


class Maps:
    sketches = {
        "small": np.array([
            list(' @@ '),
            list(' @  '),
            list('    ')]),
        "medium.json": np.array([
            list('     '),
            list('  @@ '),
            list('  @  '),
            list('     ')]),
        "very_large": np.array([
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
    sketches["large"] = sketches["very_large"][:, :16]
    apple_regen = {
        "small": 0.05,
        "medium.json": 0.05,
        "large": 0.005,
        "very_large": 0.005
    }

    def __init__(self, sketch, init_state="empty"):
        self.sketch = sketch
        self.apple_regen = Maps.apple_regen[sketch]
        self.init_state = init_state

        self.current_state = Maps.sketches[self.sketch]
        self.apple_cells = np.where(self.current_state == '@')
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

    def regen_apples(self):
        """
        Regenerates apples on the map with probability apple_regen. Using apple cells as a mask, we generate a random binomial
        distribution with probability apple_regen and replace the apple cells with the result.
        :return:
        """
        mask = np.random.binomial(1, self.apple_regen, size=self.apple_cells[0].shape)
        final_value = np.logical_or(mask, self.current_state[self.apple_cells] == '@')
        self.current_state[self.apple_cells] = np.where(final_value, '@', ' ')
