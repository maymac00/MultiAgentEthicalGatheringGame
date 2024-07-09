from setuptools import setup, find_packages

setup(name='MultiAgentEthicalGathering',
      version='0.1',
      install_requires=['gym', 'matplotlib', 'numpy', 'prettytable'],
      packages=find_packages(),
      )