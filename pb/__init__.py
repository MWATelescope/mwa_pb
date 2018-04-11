"""
This module defines a python interface into a model of the primary beam.

It can either be the simple analytic model or the more advanced semi-analytic model
"""

# import measured_beamformer,mwapb

import mwapy
import os

datadir = os.path.join(os.path.dirname(mwapy.__file__), 'data')

# dipole height in m
DIPOLE_HEIGHT = 0.278
# dipole separation in m
DIPOLE_SEPARATION = 1.1
# delay unit in s
DELAY_INT = 435.0e-12

# for 2014 beam model
Zmatrix = os.path.join(datadir,'ZMatrix.fits')
Jmatrix = os.path.join(datadir,'Jmatrix.fits')

# 2016 beam model
h5file = os.path.join(datadir, 'mwa_full_embedded_element_pattern.h5')
h5fileversion = "UNDEFINED"

try:
  import h5py
  h5f = h5py.File(h5file,'r')
  h5fileversion = h5f.attrs["VERSION"]
except:
  pass

