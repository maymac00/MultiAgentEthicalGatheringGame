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

# Register the presets
envs_tags= ["tiny", "small", "medium", "large", "very_large"]
envs = [presets.tiny, presets.small, presets.medium, presets.large, presets.very_large]
for tag, env in zip(envs_tags, envs):
    args = copy.copy(env)
    register(
        id=f'MultiAgentEthicalGathering-{env}-v1',
        entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG',
        kwargs= {**args}
    )
    # multi-objective version
    args["reward_mode"] = "vectorial"
    args["objective_order"] = "ethical_first"
    register(
        id=f'MultiAgentEthicalGathering-{env}-mo-v1',
        entry_point='EthicalGatheringGame.MultiAgentEthicalGathering:MAEGG',
        kwargs= {**args}
    )


