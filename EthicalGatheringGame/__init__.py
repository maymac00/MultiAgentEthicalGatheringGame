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
    id='MultiAgentEthicalGathering-very-large-v1',
    entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG',
    kwargs= {**presets.very_large}
)


