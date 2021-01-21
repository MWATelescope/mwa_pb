
MWA Telescope Primary Beam code

For more information, see http://www.mwatelescope.org

Requirements:
    -numpy
    -astropy
    -skyfield
    -matplotlib
    -scipy
    -h5py

Optional requirements for the skymap.py module, to generate all-sky maps with
radio sources, constellations, etc, and the MWA beam overlaid as a contour plot:
    -ephem  (to identify what constellation name a given RA/Dec is in)
    -Pillow (to generate PNG files)


NOTE - to use this software, you'll need to download a data file that exceeds
github's maximum file size.

After cloning this repository:
$ cd mwa_pb/mwa_pb/data
$ wget http://ws.mwatelescope.org/static/mwa_full_embedded_element_pattern.h5
$ python setup.py install
