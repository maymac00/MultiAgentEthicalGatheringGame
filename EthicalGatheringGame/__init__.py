from gym.envs.registration import register
from gym import utils

from EthicalGatheringGame.MultiAgentEthicalGathering import MAEGG
from EthicalGatheringGame.wrappers import *
import EthicalGatheringGame.presets as presets

# Register the environment
register(
    id='MultiAgentEthicalGathering-v1',
    entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG'
)

