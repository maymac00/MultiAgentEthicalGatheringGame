from gym.envs.registration import register
# Register the environment
register(
    id='MultiAgentEthicalGathering-v1',
    entry_point='env.env:MultiAgentEthicalGathering')

