#!/usr/bin/python

import io
import logging
import math
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy

import ephem   # Only used to look up what constellation a given ra/dec is in

import astropy
from astropy.coordinates import Latitude, Longitude   # Used for parsing sexagesimal strings
from astropy.table import Table
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord

import matplotlib

if 'matplotlib.backends' not in sys.modules:
    matplotlib.use('agg')
import matplotlib.pyplot as plt

try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    Basemap = None
    print("Basemap not found - install it with:")
    print("    sudo apt-get install libgeos-3.6.2 libgeos-c1v5 libgeos-dev")
    print("    pip install --user git+https://github.com/matplotlib/basemap.git")
    print("    (or 'conda install basemap' if using Anaconda Python)")

from PIL import Image

from . import config
from . import primarybeammap as primarybeammap
from . import skyfield_utils as su

logging.basicConfig()
DEFAULTLOGGER = logging.getLogger()

XMAS = False


SCALE = 1
FIGSIZE = 8
DPI = 150
LOW = 100
HIGH = 10000
CM = plt.cm.gray
CMI = CM.reversed()


def calc_delays(az=0.0, el=0.0):
    """
       Function calc_delays
       This function takes in an azimuth and zenith angle as
       inputs and creates and returns a 16-element byte array for
       delayswitches which have values corresponding to each
       dipole in the tile having a maximal coherent amplitude in the
       desired direction.

       This will return null if the inputs are
       out of physical range (if za is bigger than 90) or
       if the calculated switches for the dipoles are out
       of range of the delaylines in the beamformer.

       azimuth of 0 is north and it increases clockwise
       zenith angle is the angle down from zenith
       These angles should be given in degrees

      Layout of the dipoles on the tile:

                 N

           0   1   2   3

           4   5   6   7
      W                    E
           8   9   10  11

           12  13  14  15

                 S
    """
    dip_sep = 1.10  # dipole separations in meters
    delaystep = 435.0  # Delay line increment in picoseconds
    maxdelay = 31  # Maximum number of deltastep delays
    c = 0.000299798  # C in meters/picosecond
    dtor = math.pi / 180.0  # convert degrees to radians
    # define zenith angle
    za = 90 - el

    # Define arrays to hold the positional offsets of the dipoles
    xoffsets = [0.0] * 16  # offsets of the dipoles in the W-E 'x' direction
    yoffsets = [0.0] * 16  # offsets of the dipoles in the S-N 'y' direction
    delays = [0.0] * 16  # The calculated delays in picoseconds
    rdelays = [0] * 16  # The rounded delays in units of delaystep

    delaysettings = [0] * 16  # return values

    # Check input sanity
    if (abs(za) > 90):
        return None

        # Offsets of the dipoles are calculated relative to the
        # center of the tile, with positive values being in the north
        # and east directions

    xoffsets[0] = -1.5 * dip_sep
    xoffsets[1] = -0.5 * dip_sep
    xoffsets[2] = 0.5 * dip_sep
    xoffsets[3] = 1.5 * dip_sep
    xoffsets[4] = -1.5 * dip_sep
    xoffsets[5] = -0.5 * dip_sep
    xoffsets[6] = 0.5 * dip_sep
    xoffsets[7] = 1.5 * dip_sep
    xoffsets[8] = -1.5 * dip_sep
    xoffsets[9] = -0.5 * dip_sep
    xoffsets[10] = 0.5 * dip_sep
    xoffsets[11] = 1.5 * dip_sep
    xoffsets[12] = -1.5 * dip_sep
    xoffsets[13] = -0.5 * dip_sep
    xoffsets[14] = 0.5 * dip_sep
    xoffsets[15] = 1.5 * dip_sep

    yoffsets[0] = 1.5 * dip_sep
    yoffsets[1] = 1.5 * dip_sep
    yoffsets[2] = 1.5 * dip_sep
    yoffsets[3] = 1.5 * dip_sep
    yoffsets[4] = 0.5 * dip_sep
    yoffsets[5] = 0.5 * dip_sep
    yoffsets[6] = 0.5 * dip_sep
    yoffsets[7] = 0.5 * dip_sep
    yoffsets[8] = -0.5 * dip_sep
    yoffsets[9] = -0.5 * dip_sep
    yoffsets[10] = -0.5 * dip_sep
    yoffsets[11] = -0.5 * dip_sep
    yoffsets[12] = -1.5 * dip_sep
    yoffsets[13] = -1.5 * dip_sep
    yoffsets[14] = -1.5 * dip_sep
    yoffsets[15] = -1.5 * dip_sep

    # First, figure out the theoretical delays to the dipoles
    # relative to the center of the tile

    # Convert to radians
    azr = az * dtor
    zar = za * dtor

    for i in range(16):
        # calculate exact delays in picoseconds from geometry...
        delays[i] = (xoffsets[i] * math.sin(azr) + yoffsets[i] * math.cos(azr)) * math.sin(zar) / c

    # Find minimum delay
    mindelay = min(delays)

    # Subtract minimum delay so that all delays are positive
    for i in range(16):
        delays[i] -= mindelay

    # Now minimize the sum of the deviations^2 from optimal
    # due to errors introduced when rounding the delays.
    # This is done by stepping through a series of offsets to
    # see how the sum of square deviations changes
    # and then selecting the delays corresponding to the min sq dev.

    # Go through once to get baseline values to compare
    bestoffset = -0.45 * delaystep
    minsqdev = 0

    for i in range(16):
        delay_off = delays[i] + bestoffset
        intdel = int(round(delay_off / delaystep))

        if (intdel > maxdelay):
            intdel = maxdelay

        minsqdev += math.pow((intdel * delaystep - delay_off), 2)

    minsqdev = minsqdev / 16

    offset = (-0.45 * delaystep) + (delaystep / 20.0)
    while offset <= (0.45 * delaystep):
        sqdev = 0
        for i in range(16):
            delay_off = delays[i] + offset
            intdel = int(round(delay_off / delaystep))

            if (intdel > maxdelay):
                intdel = maxdelay
            sqdev = sqdev + math.pow((intdel * delaystep - delay_off), 2)

        sqdev = sqdev / 16
        if (sqdev < minsqdev):
            minsqdev = sqdev
            bestoffset = offset

        offset += delaystep / 20.0

    for i in range(16):
        rdelays[i] = int(round((delays[i] + bestoffset) / delaystep))
        if (rdelays[i] > maxdelay):
            if (rdelays[i] > maxdelay + 1):
                return None  # Trying to steer out of range.
            rdelays[i] = maxdelay

    # Set the actual delays
    for i in range(16):
        delaysettings[i] = int(rdelays[i])

    return delaysettings


class SkyData(object):
    def __init__(self, logger=DEFAULTLOGGER):
        su.init_data()
        self.valid = True
        # read the constellation data
        try:
            fi = open(config.CONSTELLATION_FILE)
            self.constellations = {}
            for l in fi.readlines():
                d = l.split()
                name = d[0]
                n = int(d[1])
                data = numpy.array(map(int, d[2:]))
                self.constellations[name] = [n, data]
            fi.close()
        except:
            logger.error('Could not find constellation data')
            self.valid = False

        # Read the GLEAM source list
        try:
            fi = fits.open(config.GLEAMCAT_FILE)
            self.gleamcat = []
            for i in range(len(fi[1].data.Name)):
                name, ra, dec, flux = fi[1].data.Name[i], fi[1].data.RAJ2000[i], fi[1].data.DEJ2000[i], fi[1].data.int_flux_151[i]
                self.gleamcat.append((name, ra, dec, flux))
            self.gleamcat.sort(key=lambda x:-x[3])  # Sort from brightest to dimmest
            fi.close()
        except:
            logger.error('Could not find GLEAM data')
            self.valid = False

        # read the HIP data on those stars
        try:
            self.hip = Table.read(config.HIP_CONSTELLATION_FILE, format='ascii.commented_header')
        except:
            logger.error('Could not find star data')
            self.valid = False

        # Solar system bodies to plot
        # includes size in pixels and color
        self.bodies = {su.PLANETS['SUN']:[120, 'yellow', 'Sun'],
                       su.PLANETS['JUPITER BARYCENTER']:[60, 'cyan', 'Jupiter'],
                       su.PLANETS['MOON']:[120, 'lightgray', 'Moon'],
                       su.PLANETS['MARS BARYCENTER']:[30, 'red', 'Mars'],
                       su.PLANETS['VENUS BARYCENTER']:[40, 'violet', 'Venus'],
                       su.PLANETS['SATURN BARYCENTER']:[50, 'skyblue', 'Saturn']}

        try:
            fname = os.path.join(config.RADIO_IMAGE_FILE)
            self.radio_image = fits.open(fname)

            self.basemap = self.radio_image
            bmh = self.basemap[0].header
            self.skymapra = (bmh.get('CRVAL1') + (numpy.arange(1, self.basemap[0].data[0].shape[1] + 1) - bmh.get('CRPIX1')) * bmh.get('CDELT1')) / 15.0
            self.skymapdec = bmh.get('CRVAL2') + (numpy.arange(1, self.basemap[0].data[0].shape[0] + 1) - bmh.get('CRPIX2')) * bmh.get('CDELT2')
            self.skymapRA, self.skymapDec = numpy.meshgrid(self.skymapra * 15, self.skymapdec)
        except:
            logger.error('Cannot open Haslam image')
            self.valid = False


def plot_MWAconstellations(outfile=None,
                           obsinfo=None,
                           viewgps=None,
                           observing=True,
                           showbeam=True,
                           constellations=True,
                           gleamsources=False,
                           notext=False,
                           inverse=False,
                           skydata=None,
                           background=None,
                           hidenulls=False,
                           channel=None,   # Frequency channel to use for the beam map, defaults to mean of all channels in obs.
                           xmas=XMAS,
                           plotscale=SCALE,  # A scale of 1.0 gives a 1200x1200 pixel plot
                           logger=DEFAULTLOGGER):
    if obsinfo is None:
        logger.error('Unable to find observation info')
        return None

    if skydata is None:
        skydata = SkyData()
    if not skydata.valid:
        logger.error('Unable to load star/planet data, aborting.')
        return None

    if background is None:
        background = 'transparent'

    if channel is None:
        if 0 in obsinfo['rfstreams']:
            channel = obsinfo['rfstreams'][0]['frequencies'][12]
        elif '0' in obsinfo['rfstreams']:
            channel = obsinfo['rfstreams']['0']['frequencies'][12]

    s_obstime = su.time2tai(obsinfo['starttime'])
    a_obstime = Time(obsinfo['starttime'], format='gps', scale='utc')

    if viewgps is None:
        s_viewtime = s_obstime
        a_viewtime = a_obstime
    else:
        s_viewtime = su.time2tai(viewgps)
        a_viewtime = Time(viewgps, format='gps', scale='utc')

    a_viewtime.delta_ut1_utc = 0  # We don't care about IERS tables and high precision answers
    LST_hours = a_viewtime.sidereal_time(kind='apparent', longitude=config.MWAPOS.lon)

    observer = su.S_MWAPOS.at(s_viewtime)

    mapzenith = SkyCoord(ra=skydata.skymapRA,
                         dec=skydata.skymapDec,
                         equinox='J2000',
                         unit=(astropy.units.deg, astropy.units.deg))
    mapzenith.location = config.MWAPOS
    mapzenith.obstime = a_viewtime
    altaz = mapzenith.transform_to('altaz')
    Az, Alt = altaz.az.deg, altaz.alt.deg

    fig = plt.figure(figsize=(FIGSIZE * plotscale, FIGSIZE * plotscale), dpi=DPI)
    ax1 = fig.add_subplot(1, 1, 1)

    bmap = Basemap(projection='ortho', lat_0=su.MWA_TOPO.latitude.degrees, lon_0=LST_hours.hour * 15 - 360, ax=ax1)
    nx = len(skydata.skymapra)
    ny = len(skydata.skymapdec)

    ax1.cla()

    # show the Haslam map
    tform_skymap = bmap.transform_scalar(skydata.basemap[0].data[0][:, ::-1],
                                         skydata.skymapra[::-1] * 15,
                                         skydata.skymapdec,
                                         nx, ny,
                                         masked=True)
    if inverse:
        cmap = CMI
    else:
        cmap = CM
    bmap.imshow(numpy.ma.log10(tform_skymap[:, ::-1]), cmap=cmap, vmin=math.log10(LOW), vmax=math.log10(HIGH))

    delays = []
    if showbeam:
        if not hidenulls:
            contours = [0.001, 0.1, 0.5, 0.90]
            if observing:
                beamcolor = ((0.0, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.75, 0.0), (0.0, 1.0, 0.0))
            else:
                beamcolor = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (0.75, 0.75, 0.75), (1.0, 1.0, 1.0))
        else:
            contours = [0.1, 0.5, 0.90]
            if observing:
                beamcolor = ((0.0, 0.5, 0.0), (0.0, 0.75, 0.0), (0.0, 1.0, 0.0))
            else:
                beamcolor = ((0.5, 0.5, 0.5), (0.75, 0.75, 0.75), (1.0, 1.0, 1.0))

        # If the observation is in the future, calculate what delays will be used, instead of using the recorded actual delays
        if su.tai2gps(s_obstime) > su.tai2gps(su.time2tai()) + 10:
            if 0 in obsinfo['rfstreams']:
                delays = calc_delays(az=obsinfo['rfstreams'][0]['azimuth'], el=obsinfo['rfstreams'][0]['elevation'])
            elif '0' in obsinfo['rfstreams']:
                delays = calc_delays(az=obsinfo['rfstreams']['0']['azimuth'], el=obsinfo['rfstreams']['0']['elevation'])
            else:
                delays = [33] * 16
            logger.debug("Calculated future delays: %s" % delays)
        else:
            if 0 in obsinfo['rfstreams']:
                delays = obsinfo['rfstreams'][0]['xdelays']
            elif '0' in obsinfo['rfstreams']:
                delays = obsinfo['rfstreams']['0']['xdelays']
            logger.debug("Used actual delays: %s" % delays)

        # get the primary beam
        R = primarybeammap.return_beam(Alt,
                                       Az,
                                       delays,
                                       channel * 1.28)

        # show the beam
        X, Y = bmap(skydata.skymapRA, skydata.skymapDec)
        CS = bmap.contour(bmap.xmax - X, Y, R, contours, linewidths=plotscale, colors=beamcolor)
        ax1.clabel(CS, inline=1, fontsize=10 * plotscale)

    # Find the constellation that the beam is in
    if obsinfo['ra_phase_center'] is not None:
        ra = obsinfo['ra_phase_center']
        dec = obsinfo['dec_phase_center']
    else:
        ra = obsinfo['metadata']['ra_pointing']
        dec = obsinfo['metadata']['dec_pointing']
    if (ra is not None) and (dec is not None):
        constellation = ephem.constellation((ra * math.pi / 180.0, dec * math.pi / 180.0))
    else:
        constellation = ["N/A", "N/A"]

    X0, Y0 = bmap(LST_hours.hour * 15 - 360, su.MWA_TOPO.latitude.degrees)

    if constellations:
        # plot the constellations
        ConstellationStars = []
        for c in skydata.constellations.keys():
            for i in range(0, len(skydata.constellations[c][1]), 2):
                i1 = numpy.where(skydata.hip['HIP'] == skydata.constellations[c][1][i])[0][0]
                i2 = numpy.where(skydata.hip['HIP'] == skydata.constellations[c][1][i + 1])[0][0]
                star1 = skydata.hip[i1]
                star2 = skydata.hip[i2]
                if i1 not in ConstellationStars:
                    ConstellationStars.append(i1)
                if i2 not in ConstellationStars:
                    ConstellationStars.append(i2)
                ra1, dec1 = map(numpy.degrees, (star1['RArad'], star1['DErad']))
                ra2, dec2 = map(numpy.degrees, (star2['RArad'], star2['DErad']))
                ra = numpy.array([ra1, ra2])
                dec = numpy.array([dec1, dec2])
                newx, newy = bmap(ra, dec)
                testx, testy = bmap(newx, newy, inverse=True)
                if testx.max() < 1e30 and testy.max() < 1e30:
                    bmap.plot(2 * X0 - newx, newy, 'r-', linewidth=plotscale, latlon=False)  # This bit generates an error

        # figure out the coordinates
        # and plot the stars
        ra = numpy.degrees(skydata.hip[ConstellationStars]['RArad'])
        dec = numpy.degrees(skydata.hip[ConstellationStars]['DErad'])
        m = numpy.degrees(skydata.hip[ConstellationStars]['Hpmag'])
        newx, newy = bmap(ra, dec)
        # testx, testy = bmap(newx, newy, inverse=True)
        good = (newx > bmap.xmin) & (newx < bmap.xmax) & (newy > bmap.ymin) & (newy < bmap.ymax)
        size = 60 - 15 * m
        size[size <= 15] = 15
        size[size >= 60] = 60
        bmap.scatter(bmap.xmax - newx[good], newy[good],
                     size[good] * plotscale,
                     'r',
                     edgecolor='none',
                     alpha=0.7)

    if gleamsources:
        ra = numpy.array([x[1] for x in skydata.gleamcat])
        dec = numpy.array([x[2] for x in skydata.gleamcat])
        flux = numpy.array([x[3] for x in skydata.gleamcat])
        newx, newy = bmap(ra, dec)
        # testx, testy = bmap(newx, newy, inverse=True)
        good = (newx > bmap.xmin) & (newx < bmap.xmax) & (newy > bmap.ymin) & (newy < bmap.ymax)
        size = flux / 1.0
        size[size <= 7] = 7
        size[size >= 60] = 60
        bmap.scatter(bmap.xmax - newx[good], newy[good],
                     size[good] * plotscale,
                     'b',
                     edgecolor='none',
                     alpha=0.7)

    # plot the bodies
    for b in skydata.bodies.keys():
        name = skydata.bodies[b][2]
        color = skydata.bodies[b][1]
        if inverse:
            if name == 'Moon':
                color = 'darkgoldenrod'
            elif name == 'Jupiter':
                color = 'sienna'
            elif name == 'Saturn':
                color = 'purple'
        size = skydata.bodies[b][0]
        body_app = observer.observe(b).apparent()
        body_ra_a, body_dec_a, _ = body_app.radec()
        ra, dec = body_ra_a._degrees, body_dec_a.degrees
        newx, newy = bmap(ra, dec)
        testx, testy = bmap(newx, newy, inverse=True)
        if testx < 1e30 and testy < 1e30:
            bmap.scatter(2 * X0 - newx, newy, s=size * plotscale, c=color, alpha=1.0, latlon=False, edgecolor='none')
            ax1.text(bmap.xmax - newx + 2e5, newy,
                     name,
                     horizontalalignment='left',
                     fontsize=12 * plotscale,
                     color=color,
                     verticalalignment='center')

    # and label some sources
    for source in primarybeammap.sources.keys():
        if source == 'EOR0b':
            continue
        if source == 'CenA':
            primarybeammap.sources[source][0] = 'Cen A'
        if source == 'ForA':
            primarybeammap.sources[source][0] = 'For A'
        r = Longitude(angle=primarybeammap.sources[source][1], unit=astropy.units.hour).hour
        d = Latitude(angle=primarybeammap.sources[source][2], unit=astropy.units.deg).deg
        horizontalalignment = 'left'
        x = r
        if (len(primarybeammap.sources[source]) >= 6 and primarybeammap.sources[source][5] == 'c'):
            horizontalalignment = 'center'
            x = r
        if (len(primarybeammap.sources[source]) >= 6 and primarybeammap.sources[source][5] == 'r'):
            horizontalalignment = 'right'
            x = r
        fontsize = primarybeammap.defaultsize
        if (len(primarybeammap.sources[source]) >= 5):
            fontsize = primarybeammap.sources[source][4]
        color = primarybeammap.defaultcolor
        if (len(primarybeammap.sources[source]) >= 4):
            color = primarybeammap.sources[source][3]
        if color == 'k':
            color = 'w'

        if inverse:
            if color == 'w':
                color = 'black'

        xx, yy = bmap(x * 15 - 360, d)
        try:
            if xx < 1e30 and yy < 1e30:
                ax1.text(bmap.xmax - xx + 2e5, yy,
                         primarybeammap.sources[source][0],
                         horizontalalignment=horizontalalignment,
                         fontsize=fontsize * plotscale,
                         color=color,
                         verticalalignment='center')
        except:
            pass

    if not notext:
        if background == 'black':
            fontcolor = 'white'
        else:
            fontcolor = 'black'

        if showbeam:
            ax1.text(0, bmap.ymax - 2e5,
                     'Obs ID %d with delays %s\n at %s:\n%s at %d MHz\n in the constellation %s' % (obsinfo['starttime'],
                                                                                                    delays,
                                                                                                    a_viewtime.datetime.strftime('%Y-%m-%d %H:%M UT'),
                                                                                                    obsinfo['obsname'],
                                                                                                    channel * 1.28,
                                                                                                    constellation[1]),
                     fontsize=10 * plotscale,
                     color=fontcolor)
        else:
            ax1.text(0, bmap.ymax - 2e5,
                     '%s:\nNo recent observation' % (a_obstime.datetime.strftime('%Y-%m-%d %H:%M UT')),
                     fontsize=10 * plotscale,
                     color=fontcolor)

    ax1.text(bmap.xmax, Y0, 'W', fontsize=12 * plotscale, horizontalalignment='left', verticalalignment='center')
    ax1.text(bmap.xmin, Y0, 'E', fontsize=12 * plotscale, horizontalalignment='right', verticalalignment='center')
    ax1.text(X0, bmap.ymax, 'N', fontsize=12 * plotscale, horizontalalignment='center', verticalalignment='bottom')
    ax1.text(X0, bmap.ymin, 'S', fontsize=12 * plotscale, horizontalalignment='center', verticalalignment='top')

    try:
        if type(outfile) == str:
            if not xmas:
                if background.lower() == 'transparent':
                    fig.savefig(outfile, transparent=True, facecolor='none', dpi=DPI)
                else:
                    fig.savefig(outfile, transparent=False, facecolor=background, dpi=DPI)
                return ''
            else:
                buf = io.BytesIO()
                if background.lower() == 'transparent':
                    fig.savefig(buf, format='png', transparent=True, facecolor='none', dpi=DPI)
                else:
                    fig.savefig(buf, format='png', transparent=False, facecolor=background, dpi=DPI)
                buf.seek(0)
                im = Image.open(buf)
                im.load()
                buf.close()
                r = Image.open('Treindeers.png')
                im.paste(r, box=(250, 300), mask=r)
                buf2 = io.BytesIO()
                im.save(buf2, format=outfile[outfile.find('.') + 1:].upper())
                buf2.seek(0)
                outf = open(outfile, 'wb')
                outf.write(buf2.read())
                return ''
        else:
            buf = io.BytesIO()
            if background.lower() == 'transparent':
                fig.savefig(buf, format='png', transparent=True, facecolor='none', dpi=DPI)
            else:
                fig.savefig(buf, format='png', transparent=False, facecolor=background, dpi=DPI)
            buf.seek(0)
            if not xmas:
                return buf.read()
            else:
                im = Image.open(buf)
                im.load()
                buf.close()
                r = Image.open('Treindeers.png')
                im.paste(r, box=(250, 100), mask=r)
                buf2 = io.BytesIO()
                im.save(buf2, format='PNG')
                buf2.seek(0)
                return buf2.read()
    except AssertionError:
        logger.error('Cannot save output: %s', outfile)
        return None
    finally:
        plt.close(fig)
        del ax1
        del fig
