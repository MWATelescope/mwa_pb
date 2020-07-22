"""
This module defines a python interface into a model of the primary beam.

It can either be the simple analytic model or the more advanced semi-analytic model
"""

# import measured_beamformer,mwapb

import os

from astropy.coordinates import EarthLocation


datadir = os.path.join(os.path.dirname(__file__), 'data')

# 2016 beam model
h5file = os.path.join(datadir, 'mwa_full_embedded_element_pattern.h5')
h5fileversion = "UNDEFINED"

__version__ = "1.2.0"

# dipole height in m
DIPOLE_HEIGHT = 0.278
# dipole separation in m
DIPOLE_SEPARATION = 1.1
# delay unit in s
DELAY_INT = 435.0e-12

# for 2014 beam model
Zmatrix = os.path.join(datadir, 'ZMatrix.fits')
Jmatrix = os.path.join(datadir, 'Jmatrix.fits')

# Hardcode the paths of the delay and gain files
MEAS_DELAYS = os.path.join(datadir, 'meas_delays.txt')
MEAS_GAINS = os.path.join(datadir, 'meas_gain_db.txt')

# Hardcode the paths of the data files used by skymap.py
CONSTELLATION_FILE = os.path.join(datadir, 'constellationship.fab')
GLEAMCAT_FILE = os.path.join(datadir, 'G4Jy_catalogue_allEGCcolumns.fits')
HIP_CONSTELLATION_FILE = os.path.join(datadir, 'HIP_constellations.dat')

# Haslam image:
RADIO_IMAGE_FILE = os.path.join(datadir, 'radio408.RaDec.fits')

MWAPOS = EarthLocation.from_geodetic(lon="116:40:14.93", lat="-26:42:11.95", height=377.8)
