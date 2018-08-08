from setuptools import setup

setup(
  name='mwa_pb',
  version='1.1.0',
  packages=['mwa_pb'],
  package_data={'mwa_pb':['data/*.fits', 'data/*.txt', 'data/*.h5']},
  url='https://github.com/MWATelescope/mwa_pb',
  license='GPLv3',
  author='MWA Team members, repo managed by Andrew Williams',
  author_email='Andrew.Williams@curtin.edu.au',
  description='MWA Primary beam code',
  scripts=['scripts/beam_correct_image.py',
           'scripts/beamtest.py',
           'scripts/calc_jones.py',
           'scripts/make_beam_test.py',
           'scripts/mwa_sensitivity.py',
           'scripts/primarybeammap_tant_test.py',
           'scripts/track_and_suppress.py'],
  install_requires=["numpy", "astropy", "matplotlib", "scipy", "ephem", "h5py"]
)
