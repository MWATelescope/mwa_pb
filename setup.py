import os
from setuptools import setup
from subprocess import check_output

#The following two functions were taken from the repo: https://github.com/pyfidelity/setuptools-git-version/blob/master/setuptools_git_version.py
def format_version(version, fmt='{tag}.{commitcount}'):
    parts = version.split('-')
    if len(parts) == 1:
        return parts[0]
    assert len(parts) in (3, 4)
    dirty = len(parts) == 4
    tag, count, sha = parts[:3]
    if count == '0' and not dirty:
        return tag
    return fmt.format(tag=tag, commitcount=count)

def get_git_version():
    git_version = check_output('git describe --tags --long --dirty --always'.split()).decode('utf-8').strip()
    return format_version(version=git_version)

# Download the mwa_full_embedded_element_pattern.h5 file if it doesn't exist
datadir = os.path.join(os.path.dirname(__file__), 'mwa_pb', 'data')
h5file = os.path.join(datadir, 'mwa_full_embedded_element_pattern.h5')
print(h5file)
if not os.path.exists(h5file):
    # Importing download functions here to avoid unnessiary imports when the file is available
    import urllib.request
    print("The mwa_full_embedded_element_pattern.h5 file does not exist. Downloading it from http://ws.mwatelescope.org")
    response = urllib.request.urlopen("http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5", timeout = 5)
    content = response.read()
    f = open(h5file, 'wb' )
    f.write( content )
    f.close()
    print("Download complete")


setup(
    name='mwa_pb',
    version=get_git_version(),
    packages=['mwa_pb'],
    package_data={'mwa_pb':['data/*.fits', 'data/*.txt', 'data/*.h5', 'data/*.fab', 'data/*.dat']},
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
             'scripts/plot_skymap.py',
             'scripts/primarybeammap_tant_test.py',
             'scripts/track_and_suppress.py'],
    install_requires=["numpy", "astropy", "skyfield>=1.16", "matplotlib", "scipy>=0.15.1", "h5py"],
    extras_require={'skymap':["ephem", "Pillow"]}   # Needed only to generate sky maps in mwa_pb/skymap.py
)
