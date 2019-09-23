"""
  Script to calculate MWA antenna temperature using one of the beam models (analytic, AEE or FEE) and scaled HASLAM sky map

  main task is:
  make_primarybeammap()
"""

import logging
import math
import os

import astropy
import astropy.io.fits as pyfits
import astropy.coordinates    # Just to use the Angle class to parse 'DD:MM:SS.sss' values

import numpy

import matplotlib

import skyfield.api as si

matplotlib.use('agg')
from matplotlib import pyplot as pylab

from scipy.interpolate import RegularGridInterpolator

import config
import beam_tools
import primary_beam
import skyfield_utils as su

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
SOURCES = {
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
    print('Using standard orthographic projection')
    az_grid = numpy.arctan2(c, r)
    mask[p] = 1.0  # set mask
    p = (dsqu >= (1.0 + EPS))
    za_grid[p] = math.pi / 2.0  # set ZA outside of fov to 90 deg
    return az_grid * 180.0 / math.pi, za_grid * 180.0 / math.pi


def map_sky(skymap, RA, dec, gps, az_grid, za_grid):
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
    print('grid2eq', grid2eq['RA'].shape)

    # Set up interp function using sky map
    # flip so interpolation has increasing values
    # TO DO: I don't think this will affect outcome!
    my_interp_fn = RegularGridInterpolator((dec, RA[::-1]), skymap[:, ::-1], fill_value=None)
    # fill_value = None means that values outside domain are extrapolated.
    # fill_value=nan would be preferable, but this causes error due to bug in scipy<15.0, as per
    # https://github.com/scipy/scipy/issues/3703

    # interpolate map onto az,ZA grid
    print(numpy.min(grid2eq['dec']), numpy.max(grid2eq['dec']))
    print(numpy.min(grid2eq['RA']), numpy.max(grid2eq['RA']))
    # Convert to RA=-180 - 180 format (same as Haslam)
    # We do it this way so RA values are always increasing for RegularGridInterpolator
    grid2eq['RA'][grid2eq['RA'] > 180] = grid2eq['RA'][grid2eq['RA'] > 180] - 360

    print(numpy.min(grid2eq['dec']), numpy.max(grid2eq['dec']))
    print(numpy.min(grid2eq['RA']), numpy.max(grid2eq['RA']))
    my_map = my_interp_fn(numpy.dstack([grid2eq['dec'], grid2eq['RA']]))
    #    print "np.vstack([grid2eq['dec'], grid2eq['RA']])",np.vstack([grid2eq['dec'], grid2eq['RA']]).shape
    #    print "np.hstack([grid2eq['dec'], grid2eq['RA']])",np.hstack([grid2eq['dec'], grid2eq['RA']]).shape
    #    print "np.dstack([grid2eq['dec'], grid2eq['RA']])",np.dstack([grid2eq['dec'], grid2eq['RA']]).shape

    return my_map


def horz2eq(az, ZA, gps):
    """
      Convert from horizontal (az, ZA) to equatorial (RA, dec)"
      Returns RA, dec,
      Inputs:
      time - GPS time
    """
    t = su.time2tai(gps)
    observer = su.S_MWAPOS.at(t)
    # logger.info('Calculating az, ZA at time %s', t.utc_iso())
    coords = observer.from_altaz(alt_degrees=(90 - ZA), az_degrees=az, distance=si.Distance(au=9e90))
    ra_a, dec_a, _ = coords.radec()
    return {'RA':ra_a._degrees, 'dec':dec_a.degrees}


def get_Haslam(freq, scaling=-2.55):
    """
      get the Haslam 408 MHz map.
      Outputs
      RA - RA in degrees (-180 - 180)
      dec - dec in degrees
    """
    radio_image_touse = config.RADIO_IMAGE_FILE
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


def get_LST(obstime):
    t = su.time2tai(obstime)
    lst = t.gast + (su.MWA_TOPO.longitude.degrees / 15)
    return lst  # keep as decimal hr


def make_primarybeammap(gps, delays, frequency, model, extension='png',
                        plottype='beamsky', figsize=14, directory=None, resolution=1000, zenithnorm=True,
                        b_add_sources=False):
    """
    """
    print("Output beam file resolution = %d , output directory = %s" % (resolution, directory))
    #    (az_grid, za_grid) = beam_tools.makeAZZA(resolution,'ZEA') #Get grids in radians
    (az_grid, za_grid, n_total, dOMEGA) = beam_tools.makeAZZA_dOMEGA(resolution, 'ZEA')  # TEST SIN vs. ZEA
    az_grid = az_grid * 180 / math.pi
    za_grid = za_grid * 180 / math.pi
    # az_grid+=180.0
    alt_grid = 90 - (za_grid)
    obstime = su.time2tai(gps)

    # first go from altitude to zenith angle
    theta = (90 - alt_grid) * math.pi / 180
    phi = az_grid * math.pi / 180

    beams = {}
    # this is the response for XX and YY
    if model == 'analytic' or model == '2014':
        # Handles theta and phi as floats, 1D, or 2D arrays (and probably higher dimensions)
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_analytic(theta, phi,
                                                                  freq=frequency, delays=delays,
                                                                  zenithnorm=zenithnorm, power=True)
    elif model == 'avg_EE' or model == 'advanced' or model == '2015' or model == 'AEE':
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_advanced(theta, phi,
                                                                  freq=frequency, delays=delays,
                                                                  power=True)
    elif model == 'full_EE' or model == '2016' or model == 'FEE' or model == 'Full_EE':
        # model_ver = '02'
        # h5filepath = 'MWA_embedded_element_pattern_V' + model_ver + '.h5'
        beams['XX'], beams['YY'] = primary_beam.MWA_Tile_full_EE(theta, phi,
                                                                 freq=frequency, delays=delays,
                                                                 zenithnorm=zenithnorm, power=True)
    # elif model == 'full_EE_AAVS05':
    #    #        h5filepath='/Users/230255E/Temp/_1508_Aug/embedded_element/h5/AAVS05_embedded_element_02_rev0.h5'
    #    # h5filepath = 'AAVS05_embedded_element_02_rev0.h5'
    #    beams['XX'], beams['YY'] = primary_beam.MWA_Tile_full_EE(theta, phi,
    #                                                             freq=frequency, delays=delays,
    #                                                             zenithnorm=zenithnorm, power=True)

    pols = ['XX', 'YY']

    # Get Haslam and interpolate onto grid
    my_map = get_Haslam(frequency)
    mask = numpy.isnan(za_grid)
    za_grid[numpy.isnan(za_grid)] = 90.0  # Replace nans as they break the interpolation
    sky_grid = map_sky(my_map['skymap'], my_map['RA'], my_map['dec'], gps, az_grid, za_grid)
    sky_grid[mask] = numpy.nan  # Remask beyond the horizon

    # test:
    # delays1 = numpy.array([[6, 6, 6, 6,
    #                        4, 4, 4, 4,
    #                        2, 2, 2, 2,
    #                        0, 0, 0, 0],
    #                       [6, 6, 6, 6,
    #                        4, 4, 4, 4,
    #                        2, 2, 2, 2,
    #                        0, 0, 0, 0]],
    #                      dtype=numpy.float32)
    # za_delays = {'0': delays1 * 0, '14': delays1, '28': delays1 * 2}
    # tile = mwa_tile.get_AA_Cached()
    # za_delay = '0'
    # (ax0, ay0) = tile.getArrayFactor(az_grid, za_grid, frequency, za_delays[za_delay])
    # val = numpy.abs(ax0)
    # val_max = numpy.nanmax(val)
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
        print('frequency=%.2f , polarisation=%s' % (frequency, pol))
        beam = beams[pol]
        beamsky = beam * sky_grid
        beam_dOMEGA = beam * dOMEGA
        print('sum(beam)', numpy.nansum(beam))
        print('sum(beamsky)', numpy.nansum(beamsky))
        beamsky_sum = numpy.nansum(beamsky)
        beam_sum = numpy.nansum(beam)
        beam_dOMEGA_sum = numpy.nansum(beam_dOMEGA)
        Tant = numpy.nansum(beamsky) / numpy.nansum(beam)
        print('Tant=sum(beamsky)/sum(beam)=', Tant)

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
                                                                                                                  float(numpy.nanmax(beamsky)))
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


def get_beam_power(delays, frequency, model, pointing_az_deg=0, pointing_za_deg=0, zenithnorm=True):
    """
    """
    if type(pointing_za_deg) is list:
        # Convert to a numpy array, then into radians
        theta = numpy.array(pointing_za_deg) * math.pi / 180
    else:
        theta = pointing_za_deg * math.pi / 180

    if type(pointing_az_deg) is list:
        # Convert to a numpy array, then into radians
        phi = numpy.array(pointing_az_deg) * math.pi / 180
    else:
        phi = pointing_az_deg * math.pi / 180

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
    # elif model == 'full_EE_AAVS05':
    #     #        h5filepath='/Users/230255E/Temp/_1508_Aug/embedded_element/h5/AAVS05_embedded_element_02_rev0.h5'
    #     # h5filepath = 'AAVS05_embedded_element_02_rev0.h5'
    #     beams['XX'], beams['YY'] = primary_beam.MWA_Tile_full_EE(theta, phi,
    #                                                              freq=frequency, delays=delays,
    #                                                              zenithnorm=zenithnorm, power=True)
    return beams


def add_sources(fig, ax1, ax2, obstime=None, az_grid=None, za_grid=None, beamsky=None):
    """Note that this function does nothing, apart from printing some coordinates.
    """
    obstime = su.time2tai(obstime)
    lst = get_LST(obstime)

    print("------------------------------")
    print("Adding sources for lst=%.2f [hours] , coordinates = (%.4f,%.4f) [deg]:" % (lst,
                                                                                      su.MWA_TOPO.longitude.degrees,
                                                                                      su.MWA_TOPO.latitude.degrees))
    print("------------------------------")
    # add text for sources
    # lst=get_LST(gps)

    for source in SOURCES:
        # Using astropy.Angle purely to convert the sexagesimal strings, because it's hard in skyfield.
        RA = astropy.coordinates.Angle(SOURCES[source][1], unit=astropy.units.hour).deg
        Dec = astropy.coordinates.Angle(SOURCES[source][2], unit=astropy.units.deg).deg
        coords = si.Star(ra_hours=RA / 15.0, dec_degrees=Dec)
        observer = su.S_MWAPOS.at(obstime)
        coords_alt, coords_az, _ = observer.observe(coords).apparent().altaz()
        az, alt = coords_az.degrees, coords_alt.degrees
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

            print("MAX(beam) = %.2f at (x,y) = (%d,%d)" % (max_beam, max_beam_x, max_beam_y))

        fstring = "%s : (%s,%s) -> (%.4f,%.4f) [deg] -> (az,za) = (%.4f,%.4f) [deg] -> (x,y) = (%d,%d)"
        params = (source, SOURCES[source][1], SOURCES[source][2], RA, Dec, az, za, x_best, y_best)
        print(fstring % params)

    print("------------------------------")


def plot_beamsky(beamsky, frequency, textlabel, filename, extension,
                 figsize=8, vmax=None, cbar_label='beam x Tsky (K)',
                 directory=None, dec=-26.7033, obstime=None,
                 b_add_sources=False, az_grid=None, za_grid=None):
    # do the plotting
    # this sets up the figure with the right aspect ratio
    obstime = su.time2tai(obstime)
    lst = get_LST(obstime)

    fig = pylab.figure(figsize=(figsize, 0.6 * figsize), dpi=300)
    pylab.axis('on')
    ax1 = fig.add_subplot(1, 1, 1, polar=False)

    pylab.axis('off')
    # Add polar grid on top (but transparent background)
    # TO DO: change grid labels to ZA.
    ax2 = fig.add_subplot(1, 1, 1, polar=True, frameon=False)
    ax2.set_theta_zero_location("N")
    ax2.set_theta_direction(-1)
    ax2.patch.set_alpha(0.0)
    ax2.tick_params(color='0.5', labelcolor='0.5')
    for spine in list(ax2.spines.values()):
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
    except RuntimeError as err:
        logger.error('Error saving figure: %s\n' % err)
        return None

    # save fits files:
    full_filename = filename + '.fits'
    if directory is not None:
        full_filename = directory + '/' + filename + '.fits'
    print("Filename2 = %s" % filename)
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
        hdulist.writeto(full_filename, overwrite=True)
        print("Saved output image to file %s" % full_filename)
    except RuntimeError as err:
        logger.error('Error saving figure: %s\n' % err)
        return None

    pylab.close()
