from setuptools import setup

setup(
  name='mwa_pb',
  version='1.1.0',
  packages=['mwa_pb'],
  url='https://github.com/MWATelescope/mwa_pb',
  license='GPLv3',
  author='MWA Team members, repo managed by Andrew Williams',
  author_email='Andrew.Williams@curtin.edu.au',
  description='MWA Primary beam code',
  scripts=[],
  install_requires=["numpy", "astropy", "matplotlib"]
)
