from setuptools import setup

setup(
  name='mwa_pb',
  version='1.1.0',
  packages=['mwa_pb'],
  package_data={'mwa_pb':['data/*.fits', 'data/*.txt']},
  url='https://github.com/MWATelescope/mwa_pb',
  license='GPLv3',
  author='MWA Team members, repo managed by Andrew Williams',
  author_email='Andrew.Williams@curtin.edu.au',
  description='MWA Primary beam code',
  scripts=['beamtest.py', 'make_beam_test.py', 'mwa_sensitivity.py', 'primarybeammap_tant_test.py', 'track_and_suppress.py'],
  install_requires=["numpy", "astropy", "matplotlib", "scipy", "ephem", "h5py"]
)
