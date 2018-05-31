#!/usr/bin/env python
"""
  Script to calculate MWA antenna temperature using one of the beam models (analytic, AEE or FEE) and scaled HASLAM sky map

  main task is:
  make_primarybeammap()
"""

import logging
import math
import os

import astropy.io.fits as pyfits

import numpy

import matplotlib

matplotlib.use('agg')
from matplotlib import pyplot as pylab

from scipy.interpolate import RegularGridInterpolator

import astropy
from astropy.time import Time
from astropy.coordinates import SkyCoord, Angle, AltAz, ICRS

import config
import beam_tools
import primary_beam
import mwa_tile

EPS = numpy.finfo(numpy.float64).eps  # machine epsilon

defaultcolor = 'k'
defaultsize = 8
contourlevels = [0.01, 0.1, 0.25, 0.5, 0.75]

# configure the logging
logging.basicConfig(format='# %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('primarybeammap')
logger.setLevel(logging.WARNING)

# information for the individual sources to label
# for each, give the name, RA, Dec, color, fontsize, and justification
# if the last three are omitted they will use the defaults
sources = {
    'HydA': ['Hyd A', '09:18:05.65', '-12:05:43.9', 'b', 10, 'l'],
    'ForA': ['For A (double)', '03:22:41.52', '-37:12:33.5', defaultcolor, defaultsize, 'l'],
    'PicA': ['Pic A', '05:19:49.73', '-45:46:43.7', 'b', 10, 'l'],
    'EOR0': ['EoR0', '00:00:00', '-27:00:00', 'w', 12, 'c'],
    'EOR0b': ['EoR0', '23:59:00', '-27:00:00', 'w', 12, 'c'],
    'EOR1': ['EoR1', '04:00:00', '-30:00:00', 'b', 12, 'c'],
    'EOR2': ['EoR2', '10:20:00', '-10:00:00', 'b', 12, 'c'],
    'PupA': ['Pup A\n(resolved)', '08:24:07', '-42:59:48'],
    '3C161': ['3C 161', '06:27:10.09', '-05:53:04.7', defaultcolor, defaultsize, 'r'],
    'M42': ['M42/Orion', '05:35:17.3', '-05:23:28'],
    'CasA': ['Cas A', '23:23:24', '+58:48:54'],
    'CygA': ['Cyg A', '19:59:28.36', '+40:44:02.1'],
    '3C444': ['3C 444', '22:14:25.75', '-17:01:36.3'],
    'PKS0408': ['PKS 0408-65', '04:08:20.37884', '-65:45:09.0806'],
    'PKS0410': ['PKS 0410-75', '04:08:48.4924', '-75:07:19.327'],
    'LMC': ['LMC', '05:23:34.6', '-69:45:22'],
    'PKS2104': ['PKS 2104-25', '21:07:25.7', '-25:25:46'],
    'PKS2153': ['PKS 2153-69', '21:57:05.98061', '-69:41:23.6855'],
    'PKS 1932': ['PKS 1932-46', '19:35:56.5', '-46:20:41', 'w'],
    'PKS1814': ['PKS 1814-63', '18:19:35.00241', '-63:45:48.1926'],
    'PKS1610': ['PKS 1610-60', '16:15:03.864', '-60:54:26.14', 'w'],
    'CenB': ['Cen B', '13:46:49.0432', '-60:24:29.355', 'w'],
    'CenA': ['Cen A (resolved)', '13:25:27.61507', '-43:01:08.8053'],
    '3C310': ['3C 310', '15:04:57.108', '+26:00:58.28'],
    '3C409': ['3C 409', '20:14:27.74', '+23:34:58.4', 'w'],
    '3C433': ['3C 433', '21:23:44.582', '+25:04:27.23', 'w'],
    'SgrA': ['Sgr A*', '17:45:40.0409', '-29:00:28.118', 'w'],
    'HerA': ['Her A', '16:51:08.147', '+04:59:33.32', defaultcolor, defaultsize, 'r'],
    '3C353': ['3C 353', '17:20:28.147', '-00:58:47.12'],
    '3C327': ['3C 327', '16:02:27.39', '+01:57:55.7'],
    '3C317': ['3C 317', '15:16:44.487', '+07:01:18.00', defaultcolor, defaultsize, 'r'],
    '3C298': ['3C 298', '14:19:08.1788', '+06:28:34.757', defaultcolor, defaultsize, 'r'],
    'VirA': ['Vir A/M87', '12:30:49.42338', '+12:23:28.0439', 'g', defaultsize, 'r'],
    '3C270': ['3C 270', '12:19:23.21621', '+05:49:29.6948', defaultcolor, defaultsize, 'r'],
    '3C273': ['3C 273', '12:29:06.69512', '+02:03:08.6628', defaultcolor, defaultsize, 'r'],
    'PKS2356': ['PKS 2356-61', '23:59:04.37', '-60:54:59.4'],
    'M1': ['M1/Crab', '05:34:31.93830', '+22:00:52.1758', 'g']
}


def show_source(source):
    if source in ['HydA', 'ForA', 'PicA', 'PupA', '3C161', 'M42', 'CasA',
                  'CygA', '3C444', 'PKS0408', 'PKS0410', 'HerA', 'SgrA', 'VirA']:
        return True
    else:
        return False


######################################################################
def get_azza_arrays_fov(gridsize=361, fov=180.0):
    """
      Converted from Randall Wayth's IDL code.

      Make Az,ZA arrays for a field of view (degrees)
      az array value range is -180 to 180 degrees

      gridsize is the grid size along an edge
      fov=180 degrees is the visible field of view
    """
    if fov > 180.0:
        logger.error("FOV of %s is too large. Max: 180 deg" % (fov))
        return None
    mask = numpy.zeros((gridsize, gridsize), float)
    za_grid = numpy.zeros((gridsize, gridsize), float)
    # create u,v plane (given as c, r)
    a = numpy.arange(-1, 1 + 1.0 / (gridsize - 1), 2.0 / (gridsize - 1))
    c, r = numpy.meshgrid(a, a)
    myfov = math.sin(fov / 2 * math.pi / 180)
    dsqu = (c ** 2 + r ** 2) * (myfov) ** 2
    p = (dsqu < (1.0 + EPS))
    za_grid[p] = numpy.arcsin(dsqu[p] ** 0.5)
    print 'Using standard orthographic projection'
    az_grid = numpy.arctan2(c, r)
    mask[p] = 1.0  # set mask
    p = (dsqu >= (1.0 + EPS))
    za_grid[p] = math.pi / 2.0  # set ZA outside of fov to 90 deg
    return az_grid * 180.0 / math.pi, za_grid * 180.0 / math.pi


######################################################################
def map_sky(skymap, obstime, az_grid, za_grid):
    """
      Converted from Randall Wayth's IDL code.

      Map skymap onto grid of arbitrary size
    """
    out = az_grid * 0.0  # new array for gridded sky

    grid = AltAz(az=Angle(az_grid, unit=astropy.units.deg),
                 alt=Angle(90 - za_grid, unit=astropy.units.deg),
                 obstime=obstime,
                 location=config.MWAPOS)
    grid_equatorial = grid.transform_to(ICRS)
    ra = grid_equatorial.ra.hour
    dec_grid = grid_equatorial.dec.deg

    size_dec = skymap.shape[0]
    size_ra = skymap.shape[1]
    p = za_grid < 90.0 + EPS  # array indices for visible sky

    # the following assumes RA=0 in centre
    # of the sky image and increases to the left.
    ra_index = (((36 - ra) % 24) / 24) * size_ra
    dec_index = (dec_grid / 180.0 + 0.5) * size_dec

    print ra_index.min(), ra_index.max()
    print dec_index.min(), dec_index.max()

    # select pixels of sky map, using ra and dec index values
    # rounded down to nearest index integer
    # print p
    # print numpy.rint(ra_index[p]),numpy.rint(dec_index[p])
    print numpy.rint(ra_index[p]).astype(int)
    print numpy.rint(dec_index[p]).astype(int)
    print skymap.shape
    out[p] = skymap[dec_index[p].astype(int), ra_index[p].astype(int)]
    return out


def map_sky_astropy(skymap, RA, dec, gps, az_grid, za_grid):
    """
      Reprojects Haslam map onto an input az, ZA grid.
      Inputs:
      skymap
      RA - 1D range of RAs (deg)
      dec - 1D range of decs (deg)
      gps - GPS time of observation
      az_grid - grid of azes onto which we map sky
      za_grid - grid of ZAs onto which we map sky
    """
    # Get az, ZA grid transformed to equatorial coords
    grid2eq = horz2eq(az_grid, za_grid, gps)
    print 'grid2eq', grid2eq['RA'].shape

    # Set up interp function using sky map
    # flip so interpolation has increasing values
    # TODO: I don't think this will affect outcome!
    my_interp_fn = RegularGridInterpolator((dec, RA[::-1]), skymap[:, ::-1], fill_value=None)
    # fill_value = None means that values outside domain are extrapolated.
    # fill_value=nan would be preferable, but this causes error due to bug in scipy<15.0, as per
    # https://github.com/scipy/scipy/issues/3703

    # interpolate map onto az,ZA grid
    print numpy.min(grid2eq['dec']), numpy.max(grid2eq['dec'])
    print numpy.min(grid2eq['RA']), numpy.max(grid2eq['RA'])
    # Convert to RA=-180 - 180 format (same as Haslam)
    # We do it this way so RA values are always increasing for RegularGridInterpolator
    grid2eq['RA'][grid2eq['RA'] > 180] = grid2eq['RA'][grid2eq['RA'] > 180] - 360

    print numpy.min(grid2eq['dec']), numpy.max(grid2eq['dec'])
    print numpy.min(grid2eq['RA']), numpy.max(grid2eq['RA'])
    my_map = my_interp_fn(numpy.dstack([grid2eq['dec'], grid2eq['RA']]))
    #    print "np.vstack([grid2eq['dec'], grid2eq['RA']])",np.vstack([grid2eq['dec'], grid2eq['RA']]).shape
    #    print "np.hstack([grid2eq['dec'], grid2eq['RA']])",np.hstack([grid2eq['dec'], grid2eq['RA']]).shape
    #    print "np.dstack([grid2eq['dec'], grid2eq['RA']])",np.dstack([grid2eq['dec'], grid2eq['RA']]).shape

    return my_map


def eq2horz(ra, dec, gps):
    """
      Convert from equatorial (RA, dec) to horizontal (az, ZA)"
      Returns Az (CW from North) and ZA in degrees at a given time,
      Inputs:
      time - GPS time
    """
    coords = SkyCoord(ra=ra, dec=dec, equinox='J2000', unit=astropy.units.deg)
    coords.location = config.MWAPOS

    # convert GPS to an astropy time object
    coords.obstime = Time(gps, format='gps', scale='utc')
    # get sidereal_time to reduced precision, by explicitly setting the offset of UT1 from UTC:
    # or the current time you can also get it much more precisely following the instructions in http://docs.astropy.org/en/latest/time/index.html#transformation-offsets)
    logger.warning('Using approximate sidereal time:')
    coords.obstime.delta_ut1_utc = 0

    logger.info('Calculating az, ZA at time %s', coords.obstime)
    mycoords = coords.transform_to('altaz')
    return {'Az': mycoords.az.deg, 'ZA': 90 - mycoords.alt.deg}


def horz2eq(az, ZA, gps):
    """
      Convert from horizontal (az, ZA) to equatorial (RA, dec)"
      Returns RA, dec,
      Inputs:
      time - GPS time
    """
    time = Time(gps, format='gps', scale='utc')
    # logger.info('Calculating az, ZA at time %s', coords.obstime)
    coords = SkyCoord(alt=90 - ZA, az=az,
                      obstime=time,
                      frame='altaz',
                      unit=astropy.units.deg,
                      equinox='J2000',
                      location=config.MWAPOS)

    # convert GPS to an astropy time object
    #    coords.obstime=Time(gps,format='gps',scale='utc')

    return {'RA': coords.icrs.ra.deg, 'dec': coords.icrs.dec.deg}


def get_Haslam(freq, scaling=-2.55):
    """
      get the Haslam 408 MHz map.
      Outputs
      RA - RA in degrees (-180 - 180)
      dec - dec in degrees
    """
    radio_image_touse = config.radio_image
    # radio_image_touse='/data/das4/packages/MWA_Tools/mwapy/pb/radio408.RaDec.fits'
    # radio_image_touse=radio_image
    if not os.path.exists(radio_image_touse):
        logger.error("Could not find 408 MHz image: %s\n" % (radio_image_touse))
        return None
    try:
        logger.info("Loading 408 MHz map from %s..." % radio_image_touse)
        f = pyfits.open(radio_image_touse)
    except Exception:
        logger.error("Error opening 408 MHz image: %s\n" % (radio_image_touse))
        return None

    skymap = f[0].data[0] / 10.0  # Haslam map is in 10xK
    skymap = skymap * (freq / 408.0e6) ** scaling  # Scale to frequency

    RA_1D = (f[0].header.get('CRVAL1') +
             (numpy.arange(1, skymap.shape[1] + 1) - f[0].header.get('CRPIX1')) * f[0].header.get('CDELT1'))  # /15.0
    dec_1D = (f[0].header.get('CRVAL2') +
              (numpy.arange(1, skymap.shape[0] + 1) - f[0].header.get('CRPIX2')) * f[0].header.get('CDELT2'))

    return {'skymap': skymap, 'RA': RA_1D, 'dec': dec_1D}  # RA, dec in degs

    ######################################################################


def get_LST(gps):
    time = Time(gps, format='gps', scale='utc')
    time.delta_ut1_utc = 0.0
    LST = time.sidereal_time('mean', config.MWAPOS.longitude.hour)
    return LST.hour  # keep as decimal hr


# FIXME: most the arguments in make_primarybeammap are not needed
def make_primarybeammap(gps, delays, frequency, model, extension='png',
                        plottype='beamsky', figsize=14, directory=None, resolution=1000, zenithnorm=True,
                        b_add_sources=False):
    """
    """
    print "Output beam file resolution = %d , output directory = %s" % (resolution, directory)
    #    (az_grid, za_grid) = beam_tools.makeAZZA(resolution,'ZEA') #Get grids in radians
    (az_grid, za_grid, n_total, dOMEGA) = beam_tools.makeAZZA_dOMEGA(resolution, 'ZEA')  # TEST SIN vs. ZEA
    az_grid = az_grid * 180 / math.pi
    za_grid = za_grid * 180 / math.pi
    # az_grid+=180.0
    alt_grid = 90 - (za_grid)
    obstime = Time(gps, format='gps', scale='utc')

    # first go from altitude to zenith angle
    theta = (90 - alt_grid) * math.pi / 180
    phi = az_grid * math.pi / 180

    beams = {}
    # this is the response for XX and YY
    if model == 'analytic' or model == '2014':
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_analytic(theta, phi,
                                                                  freq=frequency, delays=delays,
                                                                  zenithnorm=zenithnorm, power=True)
    elif model == 'avg_EE' or model == 'advanced' or model == '2015' or model == 'AEE':
        print "Using adanced model ???"
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_advanced(theta, phi,
                                                                  freq=frequency, delays=delays,
                                                                  power=True)
    elif model == 'full_EE' or model == '2016' or model == 'FEE' or model == 'Full_EE':
        # model_ver = '02'
        # h5filepath = 'MWA_embedded_element_pattern_V' + model_ver + '.h5'
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_full_EE(theta, phi,
                                                                 freq=frequency, delays=delays,
                                                                 zenithnorm=zenithnorm, power=True)
    elif model == 'full_EE_AAVS05':
        #        h5filepath='/Users/230255E/Temp/_1508_Aug/embedded_element/h5/AAVS05_embedded_element_02_rev0.h5'
        # h5filepath = 'AAVS05_embedded_element_02_rev0.h5'
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_full_EE(theta, phi,
                                                                 freq=frequency, delays=delays,
                                                                 zenithnorm=zenithnorm, power=True)

    pols = ['XX', 'YY']

    # Get Haslam and interpolate onto grid
    my_map = get_Haslam(frequency)
    mask = numpy.isnan(za_grid)
    za_grid[numpy.isnan(za_grid)] = 90.0  # Replace nans as they break the interpolation
    sky_grid = map_sky_astropy(my_map['skymap'], my_map['RA'], my_map['dec'], gps, az_grid, za_grid)
    sky_grid[mask] = numpy.nan  # Remask beyond the horizon

    # test:
    delays1 = numpy.array([[6, 6, 6, 6,
                            4, 4, 4, 4,
                            2, 2, 2, 2,
                            0, 0, 0, 0],
                           [6, 6, 6, 6,
                            4, 4, 4, 4,
                            2, 2, 2, 2,
                            0, 0, 0, 0]],
                          dtype=numpy.float32)
    za_delays = {'0': delays1 * 0, '14': delays1, '28': delays1 * 2}
    d = mwa_tile.Dipole('lookup')
    tile = mwa_tile.ApertureArray(dipoles=[d] * 16)
    za_delay = '0'
    (ax0, ay0) = tile.getArrayFactor(az_grid, za_grid, frequency, za_delays[za_delay])
    val = numpy.abs(ax0)
    val_max = numpy.nanmax(val)
    # print "VALUE : %.8f %.8f %.8f" % (frequency, val_max[0], val[resolution / 2, resolution / 2])

    beamsky_sum_XX = 0
    beam_sum_XX = 0
    Tant_XX = 0
    beam_dOMEGA_sum_XX = 0
    beamsky_sum_YY = 0
    beam_sum_YY = 0
    Tant_YY = 0
    beam_dOMEGA_sum_YY = 0

    for pol in pols:
        # Get gridded sky
        print 'frequency=%.2f , polarisation=%s' % (frequency, pol)
        beam = beams[pol]
        beamsky = beam * sky_grid
        beam_dOMEGA = beam * dOMEGA
        print 'sum(beam)', numpy.nansum(beam)
        print 'sum(beamsky)', numpy.nansum(beamsky)
        beamsky_sum = numpy.nansum(beamsky)
        beam_sum = numpy.nansum(beam)
        beam_dOMEGA_sum = numpy.nansum(beam_dOMEGA)
        Tant = numpy.nansum(beamsky) / numpy.nansum(beam)
        print 'Tant=sum(beamsky)/sum(beam)=', Tant

        if pol == 'XX':
            beamsky_sum_XX = beamsky_sum
            beam_sum_XX = beam_sum
            Tant_XX = Tant
            beam_dOMEGA_sum_XX = beam_dOMEGA_sum

        if pol == 'YY':
            beamsky_sum_YY = beamsky_sum
            beam_sum_YY = beam_sum
            Tant_YY = Tant
            beam_dOMEGA_sum_YY = beam_dOMEGA_sum

        filename = '%s_%.2fMHz_%s_%s' % (gps, frequency / 1.0e6, pol, model)
        fstring = "%.2f" % (frequency / 1.0e6)

        if plottype == 'all':
            plottypes = ['beam', 'sky', 'beamsky', 'beamsky_scaled']
        else:
            plottypes = [plottype]

        for pt in plottypes:
            if pt == 'beamsky':
                textlabel = 'Beam x sky %s (LST %.2f hr), %s MHz, %s-pol, Tant=%.1f K' % (gps,
                                                                                          get_LST(gps),
                                                                                          fstring,
                                                                                          pol,
                                                                                          Tant)
                plot_beamsky(beamsky, frequency, textlabel, filename, extension,
                             obstime=obstime, figsize=figsize, directory=directory)
            elif pt == 'beamsky_scaled':
                textlabel = 'Beam x sky (scaled) %s (LST %.2f hr), %s MHz, %s-pol, Tant=%.1f K (max T=%.1f K)' % (gps,
                                                                                                                  get_LST(gps),
                                                                                                                  fstring,
                                                                                                                  pol,
                                                                                                                  Tant,
                                                                                                                  numpy.nanmax(beamsky)[0])
                plot_beamsky(beamsky, frequency, textlabel, filename + '_scaled', extension,
                             obstime=obstime, figsize=figsize, vmax=numpy.nanmax(beamsky) * 0.4, directory=directory)

            elif pt == 'beam':
                textlabel = 'Beam for %s, %s MHz, %s-pol' % (gps, fstring, pol)
                plot_beamsky(beam, frequency, textlabel, filename + '_beam', extension,
                             obstime=obstime, figsize=figsize, cbar_label='', directory=directory,
                             b_add_sources=b_add_sources,
                             az_grid=az_grid, za_grid=za_grid)
            elif pt == 'sky':
                textlabel = 'Sky for %s (LST %.2f hr), %s MHz, %s-pol' % (gps, get_LST(gps), fstring, pol)
                plot_beamsky(sky_grid, frequency, textlabel, filename + '_sky', extension,
                             obstime=obstime, figsize=figsize, directory=directory, b_add_sources=b_add_sources,
                             az_grid=az_grid, za_grid=za_grid)

    return (beamsky_sum_XX,
            beam_sum_XX,
            Tant_XX,
            beam_dOMEGA_sum_XX,
            beamsky_sum_YY,
            beam_sum_YY,
            Tant_YY,
            beam_dOMEGA_sum_YY)


# FIXME: most the arguments in make_primarybeammap are not needed
def get_beam_power(gps, delays, frequency, model, pointing_az_deg=0, pointing_za_deg=0, zenithnorm=True):
    """
    """
    # lst = get_LST(gps)

    # first go from altitude to zenith angle
    theta_rad = pointing_za_deg * math.pi / 180
    phi_rad = pointing_az_deg * math.pi / 180

    theta = numpy.array([[theta_rad]])
    phi = numpy.array([[phi_rad]])

    beams = {}
    # this is the response for XX and YY
    if model == 'analytic' or model == '2014':
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_analytic(theta, phi,
                                                                  freq=frequency, delays=delays,
                                                                  zenithnorm=zenithnorm, power=True)
    elif model == 'avg_EE' or model == 'advanced' or model == '2015':
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_advanced(theta, phi,
                                                                  freq=frequency, delays=delays,
                                                                  power=True)
    elif model == 'full_EE' or model == '2016' or model == 'FEE' or model == 'Full_EE':
        # model_ver = '02'
        # h5filepath = 'MWA_embedded_element_pattern_V' + model_ver + '.h5'
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_full_EE(theta, phi,
                                                                 freq=frequency, delays=delays,
                                                                 zenithnorm=zenithnorm, power=True)
    elif model == 'full_EE_AAVS05':
        #        h5filepath='/Users/230255E/Temp/_1508_Aug/embedded_element/h5/AAVS05_embedded_element_02_rev0.h5'
        # h5filepath = 'AAVS05_embedded_element_02_rev0.h5'
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_full_EE(theta, phi,
                                                                 freq=frequency, delays=delays,
                                                                 zenithnorm=zenithnorm, power=True)
    return beams


def add_sources(fig, ax1, ax2, obstime=None, az_grid=None, za_grid=None, beamsky=None):
    """Note that this function does nothing, apart from printing some coordinates.
    """
    obstime.location = config.MWAPOS
    obstime.delta_ut1_utc = 0
    lst = obstime.sidereal_time(kind='mean').hour

    print "------------------------------"
    print "Adding sources for lst=%.2f [hours] , coordinates = (%.4f,%.4f) [deg]:" % (lst,
                                                                                      config.MWAPOS.longitude.deg,
                                                                                      config.MWAPOS.latitude.deg)
    print "------------------------------"
    # add text for sources
    # lst=get_LST(gps)

    for source in sources:
        RA = Angle(sources[source][1], unit=astropy.units.hour).deg
        Dec = Angle(sources[source][2], unit=astropy.units.deg).deg
        coords = SkyCoord(ra=RA, dec=Dec, equinox='J2000', unit=(astropy.units.deg, astropy.units.deg))
        coords.location = config.MWAPOS
        coords.obstime = obstime
        coords_prec = coords.transform_to('altaz')
        az, alt = coords_prec.az.deg, coords_prec.alt.deg
        za = 90.00 - alt

        x_best = -1
        y_best = -1
        if az_grid is not None and za_grid is not None and alt > 0 and show_source(source):
            min_diff = 1000000.00
            max_beam = -100000
            max_beam_x = -1
            max_beam_y = -1
            for x in range(0, az_grid.shape[0]):
                for y in range(0, az_grid.shape[1]):
                    az_test = az_grid[x, y]
                    za_test = za_grid[x, y]

                    diff = ((az - az_test) ** 2 + (za - za_test) ** 2)
                    if diff < min_diff:
                        min_diff = diff
                        x_best = x
                        y_best = y
                    if beamsky is not None:
                        if beamsky[x, y] > max_beam:
                            max_beam = beamsky[x, y]
                            max_beam_x = x
                            max_beam_y = y

            tmp = x_best
            x_best = y_best
            y_best = tmp

            tmp = max_beam_x
            max_beam_x = max_beam_y
            max_beam_y = tmp

            print "MAX(beam) = %.2f at (x,y) = (%d,%d)" % (max_beam, max_beam_x, max_beam_y)

        fstring = "%s : (%s,%s) -> (%.4f,%.4f) [deg] -> (az,za) = (%.4f,%.4f) [deg] -> (x,y) = (%d,%d)"
        params = (source, sources[source][1], sources[source][2], RA, Dec, az, za, x_best, y_best)
        print fstring % params

    print "------------------------------"


def plot_beamsky(beamsky, frequency, textlabel, filename, extension,
                 figsize=8, vmax=None, cbar_label='beam x Tsky (K)',
                 directory=None, dec=-26.7033, obstime=None,
                 b_add_sources=False, az_grid=None, za_grid=None):
    # do the plotting
    # this sets up the figure with the right aspect ratio
    obstime.delta_ut1_utc = 0.0
    lst = obstime.sidereal_time('mean', config.MWAPOS.longitude.hour).hour

    fig = pylab.figure(figsize=(figsize, 0.6 * figsize), dpi=300)
    pylab.axis('on')
    ax1 = fig.add_subplot(1, 1, 1, polar=False)

    pylab.axis('off')
    # Add polar grid on top (but transparent background)
    # TODO: change grid labels to ZA.
    ax2 = fig.add_subplot(1, 1, 1, polar=True, frameon=False)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.patch.set_alpha(0.0)
    ax2.tick_params(color='0.5', labelcolor='0.5')
    for spine in ax2.spines.values():
        spine.set_edgecolor('0.5')
    ax2.grid(which='major', color='0.5')

    # Beamsky example:
    if vmax is not None:
        im = ax1.imshow(beamsky, interpolation='none', vmax=vmax)
    else:
        im = ax1.imshow(beamsky, interpolation='none')
    # Add colorbar on own axis
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax, label='Tsky (K)') #Require more recent numpy (e.g. 1.9.2 works)
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(cbar_label)

    ax1.set_title(textlabel + '\n\n')

    if b_add_sources:
        add_sources(fig, ax1, ax2, obstime=obstime, az_grid=az_grid, za_grid=za_grid, beamsky=beamsky)

    full_filename = filename
    if directory is not None:
        full_filename = directory + '/' + filename
    try:
        fig.savefig(full_filename + '.' + extension)  # transparent=True if we  want transparent png
    except RuntimeError, err:
        logger.error('Error saving figure: %s\n' % err)
        return None

    # save fits files:
    full_filename = filename + '.fits'
    if directory is not None:
        full_filename = directory + '/' + filename + '.fits'
    print "Filename2 = %s" % filename
    try:
        hdu = pyfits.PrimaryHDU()

        # nan -> 0
        beamsky[numpy.isnan(beamsky)] = 0.0
        hdu.data = beamsky

        # add keywords:
        pixscale = 180.0 / (beamsky.shape[0] / 2)  # for all-sky
        # pixscale = 180.0 / (beamsky.shape[0]) # TEST
        hdu.header['CRPIX1'] = beamsky.shape[0] / 2 + 1
        hdu.header['CDELT1'] = pixscale / math.pi
        hdu.header['CRVAL1'] = lst * 15.00
        hdu.header['CTYPE1'] = 'RA---SIN'
        hdu.header['CRPIX2'] = beamsky.shape[0] / 2 + 1
        hdu.header['CDELT2'] = pixscale / math.pi
        hdu.header['CRVAL2'] = dec
        hdu.header['CTYPE2'] = 'DEC--SIN'
        hdu.header['BEAM_AZ'] = 0
        hdu.header['BEAM_ZA'] = 0
        hdu.header['FREQ'] = frequency

        hdulist = pyfits.HDUList([hdu])
        hdulist.writeto(full_filename, clobber=True)
        print "Saved output image to file %s" % full_filename
    except RuntimeError, err:
        logger.error('Error saving figure: %s\n' % err)
        return None

    pylab.close()
