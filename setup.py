from setuptools import setup, find_packages

setup(name='MultiAgentEthicalGatheringGame',
      version='0.1',
      install_requires=['gymnasium', 'pettingzoo', 'matplotlib', 'numpy', 'prettytable'],
      packages=find_packages(),
      )