import copy

from gymnasium import register
from EthicalGatheringGame.MultiAgentEthicalGathering import MAEGG
from EthicalGatheringGame.wrappers import *
import EthicalGatheringGame.presets as presets

# Register the environment

register(
    id='MultiAgentEthicalGathering-v1',
    entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG'
)

# Register the presets, and, for large preset, the db, and we

register(
    id='MultiAgentEthicalGathering-tiny-v1',
    entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG',
    kwargs= {**presets.tiny}
)

register(
    id='MultiAgentEthicalGathering-small-v1',
    entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG',
    kwargs= {**presets.small}
)

register(
    id='MultiAgentEthicalGathering-medium-v1',
    entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG',
    kwargs= {**presets.medium}
)

register(
    id='MultiAgentEthicalGathering-large-v1',
    entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG',
    kwargs= {**presets.large}
)

p = dict(**presets.large)
for db in [0, 1, 10, 1000]:
    for we in [0, 10, 2.6]:
        for eff_rate in [0, 0.2, 0.4, 0.6, 0.8, 1]:

            p["donation_capacity"] = db
            p["we"] = [1, we]
            p["efficiency"] = [0.85] * int(5 * eff_rate) + [0.2] * int(5 - eff_rate *5)
            register(
                id=f'MultiAgentEthicalGathering-large-db{db}-eff{eff_rate}-we{we}-v1',
                entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG',
                kwargs=copy.copy(p)
            )
