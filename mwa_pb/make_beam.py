
import datetime
import logging
import os
import platform

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

import numpy

import config
import primary_beam

import astropy
from astropy.time import Time
from astropy.coordinates import SkyCoord

# configure the logging
logging.basicConfig(format='# %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)


# logger.setLevel(logging.WARNING)


######################################################################
def make_beam(filename, ext=0, delays=None,
              model='2016',
              jones=False,
              interp=True,
              dipheight=config.DIPOLE_HEIGHT,
              dip_sep=config.DIPOLE_SEPARATION):
  """
     outputfiles=make_beam(filename, ext=0, delays=None, analytic_model=False,jones=False
     dipheight=primary_beam._DIPOLE_HEIGHT,
     dip_sep=primary_beam._DIPOLE_SEPARATION)
  """

  try:
    assert model in ['analytic', '2014', '2016']
  except AssertionError:
    logging.error('Model %s is not supported' % model)
    return None

  logger.debug('Time (start): %s' % datetime.datetime.now().time())
  # if jones and model=='analytic':
  #    logger.warning('Cannot compute Jones matrix for analytic model: using 2014 model...')
  #    model='2014'

  if delays is None:
    delays = [0] * 16
  if len(delays) != 16:
    logger.error('Require 16 delays but %d supplied' % len(delays))
    return None

  try:
    f = pyfits.open(filename)
  except IOError, err:
    logger.error('Unable to open %s for reading\n%s', filename, err)
    return None
  if isinstance(ext, int):
    if len(f) < ext + 1:
      logger.error('FITS file %s does not have extension %d' % (filename, ext))
      return None
  elif isinstance(ext, str):
    for extnum in xrange(len(f)):
      if ext.upper() == f[extnum].name:
        logger.info('Found matching extension %s[%d] = %s' % (filename, extnum, ext))
        ext = extnum
        break

  h = f[ext].header

  wcs = pywcs.WCS(h)

  naxes = h['NAXIS']

  if 'HPX' in h['CTYPE1']:
    logger.error('Cannot deal with HPX coordinates')
    return None

  freqfirst = True
  # try  order  RA,Dec,Freq,Stokes
  if 'RA' not in h['CTYPE1']:
    logger.error('Coordinate 1 should be RA')
    return None
  if 'DEC' not in h['CTYPE2']:
    logger.error('Coordinate 1 should be DEC')
    return None
  if 'FREQ' not in h['CTYPE3']:
    freqfirst = False
    if 'FREQ' not in h['CTYPE4']:
      logger.error('Coordinate 3 or 4 should be FREQ')
      return None
  if freqfirst:
    logger.debug('axis 3 is FREQ, axis 4 is STOKES')
    nfreq = h['NAXIS3']  # read number of frequency channels
    df = h['CDELT3']  # read frequency increment
  else:
    logger.debug('axis 3 is STOKES, axis 4 is FREQ')
    nfreq = h['NAXIS4']
    df = h['CDELT4']
  logger.info('Number of frequency channels = ' + str(nfreq))
  # construct the basic arrays
  x = numpy.arange(1, h['NAXIS1'] + 1)
  y = numpy.arange(1, h['NAXIS2'] + 1)
  # assume we want the first frequency
  # if we have a cube this will have to change
  ff = 1
  # X,Y=numpy.meshgrid(x,y)
  Y, X = numpy.meshgrid(y, x)

  Xflat = X.flatten()
  Yflat = Y.flatten()
  FF = ff * numpy.ones(Xflat.shape)
  Tostack = [Xflat, Yflat, FF]
  for i in xrange(3, naxes):
    Tostack.append(numpy.ones(Xflat.shape))
  pixcrd = numpy.vstack(Tostack).transpose()

  try:
    # Convert pixel coordinates to world coordinates
    # The second argument is "origin" -- in this case we're declaring we
    # have 1-based (Fortran-like) coordinates.
    sky = wcs.wcs_pix2world(pixcrd, 1)
  except Exception, e:
    logger.error('Problem converting to WCS: %s' % e)
    return None

  # extract the important pieces
  ra = sky[:, 0]
  dec = sky[:, 1]
  if freqfirst:
    freq = sky[:, 2]
  else:
    freq = sky[:, 3]
  freq = freq[numpy.isfinite(freq)][0]
  if nfreq > 1:
    frequencies = numpy.arange(nfreq) * df + freq
  else:
    frequencies = numpy.array([freq])

  # and make them back into arrays
  RA = ra.reshape(X.shape)
  Dec = dec.reshape(Y.shape)

  # get the date so we can convert to Az,El
  try:
    d = h['DATE-OBS']
  except KeyError:
    logger.error('Unable to read observation date DATE-OBS from %s' % filename)
    return None

  if '.' in d:
    d = d.split('.')[0]
  dt = datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
  mwatime = Time(dt)
  logger.info('Computing for %s UTC (%d)' % (mwatime.iso, mwatime.gps))
  coords = SkyCoord(ra=RA,
                    dec=Dec,
                    equinox='J2000',
                    unit=(astropy.units.deg, astropy.units.deg))
  coords.location = config.MWAPOS
  coords.obstime = mwatime
  coords_prec = coords.transform_to('altaz')
  Az, Alt = coords_prec.az.deg, coords_prec.alt.deg

  # go from altitude to zenith angle
  theta = numpy.radians((90 - Alt))
  phi = numpy.radians(Az)

  tempY = numpy.zeros(f[ext].data.shape)  # copy to prevent rY from overwriting rX in the loop below
  rX, rY, J = None, None, None
  for freqindex in xrange(len(frequencies)):
    logger.debug('Time (get beam): %s , frequency = %.2f' % (datetime.datetime.now().time(), frequencies[freqindex]))
    try:
      if model == 'analytic':
        if not jones:
          rX, rY = primary_beam.MWA_Tile_analytic(theta, phi,
                                                  freq=frequencies[freqindex],
                                                  delays=delays,
                                                  dipheight=dipheight,
                                                  dip_sep=dip_sep,
                                                  zenithnorm=True,
                                                  power=True)
        else:
          J = primary_beam.MWA_Tile_analytic(theta, phi,
                                             freq=frequencies[freqindex],
                                             delays=delays,
                                             dipheight=dipheight,
                                             dip_sep=dip_sep,
                                             zenithnorm=True,
                                             jones=True)

      elif model == '2016':
        if not jones:
          rX, rY = primary_beam.MWA_Tile_full_EE(theta, phi,
                                                 freq=frequencies[freqindex],
                                                 delays=delays,
                                                 zenithnorm=True,
                                                 power=True,
                                                 interp=interp)
        else:
          J = primary_beam.MWA_Tile_full_EE(theta, phi,
                                            freq=frequencies[freqindex],
                                            delays=delays,
                                            zenithnorm=True,
                                            jones=True,
                                            interp=interp)
      elif model == '2014':
        if not jones:
          rX, rY = primary_beam.MWA_Tile_advanced(theta, phi,
                                                  freq=frequencies[freqindex],
                                                  delays=delays,
                                                  power=True)
        else:
          J = primary_beam.MWA_Tile_advanced(theta, phi,
                                             freq=frequencies[freqindex],
                                             delays=delays,
                                             jones=True)

    except Exception, e:
      print e
      logger.error('Problem creating primary beam: %s' % e)
      return None
    logger.info('Created primary beam for %.2f MHz and delays=%s' %
                (frequencies[freqindex] / 1.0e6, ','.join([str(x) for x in delays])))
    logger.debug('Time (beam made): %s' % datetime.datetime.now().time())

    # here is the needed transposition,
    # which in full_Stokes_beam_correct_auto.py and generate_beam.py
    # is implemented in function get_azza_from_fits
    # (see return statement there)
    if not jones:
      if freqfirst:
        f[ext].data[0, freqindex] = rX.transpose()
        tempY[0, freqindex] = rY.transpose()
      else:
        f[ext].data[freqindex, 0] = rX.transpose()
        tempY[freqindex, 0] = rY.transpose()

  if 'python_version' in platform.__dict__:
    pyver = platform.python_version()
  else:
    pyver = '0.0'
  f[ext].header.set('PYVER', pyver, 'PYTHON Version number')
  pyfitsver = pyfits.__dict__.get('__version__', '0.0')
  f[ext].header.set('PYFITS', pyfitsver, 'PYFITS Version number')

  pywcsver = pywcs.__dict__.get('__version__', '0.0')
  f[ext].header.set('PYWCS', pywcsver, 'PYWCS Version number')
  f[ext].header.set('DIPHT', dipheight, '[m] Dipole height')
  f[ext].header.set('DIPSP', dip_sep, '[m] Dipole separation')
  f[ext].header.set('MWAVER', config.__version__, 'MWAPY Version')

  if model == 'analytic':
    f[ext].header.set('BEAMMODL', 'ANALYTIC', 'Primary beam model')
  elif model == '2016':
    f[ext].header.set('BEAMMODL', 'SH_V' + str(config.h5fileversion), 'Primary beam model')
  elif model == '2014':
    f[ext].header.set('BEAMMODL', 'SUTINJO14', 'Primary beam model')

  if jones:
    root = os.path.splitext(filename)[0]
    outnames = []
    # TODO: Why is this XX not XTHETA?
    pols = [['XX', -5, 0, 0],
            ['XY', -7, 0, 1],
            ['YX', -6, 1, 0],
            ['YY', -8, 1, 1]]
    for p in pols:
      if freqfirst:
        f[ext].data[0, len(frequencies) - 1] = J[:, :, p[2], p[3]].transpose().real  # Changed from 'freqindex' to 'len(frequencies) - 1' to avoid using loop index
        f[ext].header['CRVAL4'] = p[1]
      else:
        f[ext].data[len(frequencies) - 1, 0] = J[:, :, p[2], p[3]].transpose().real  # Changed from 'freqindex' to 'len(frequencies) - 1' to avoid using loop index
        f[ext].header['CRVAL3'] = p[1]
      f[ext].header.set('POLN', p[0] + '-real')
      outname = '%s_beam%sr.fits' % (root, p[0])
      if os.path.exists(outname):
        os.remove(outname)
      f[ext].writeto(outname)
      logger.info('%s-real beam written to %s' % (p[0], outname))
      outnames.append(outname)

      if freqfirst:
        f[ext].data[0, len(frequencies) - 1] = J[:, :, p[2], p[3]].transpose().imag    # Changed from 'freqindex' to 'len(frequencies) - 1' to avoid using loop index
        f[ext].header['CRVAL4'] = p[1]
      else:
        f[ext].data[len(frequencies) - 1, 0] = J[:, :, p[2], p[3]].transpose().imag    # Changed from 'freqindex' to 'len(frequencies) - 1' to avoid using loop index
        f[ext].header['CRVAL3'] = p[1]
      f[ext].header.set('POLN', p[0] + '-imag')
      outname = '%s_beam%si.fits' % (root, p[0])
      if os.path.exists(outname):
        os.remove(outname)
      f[ext].writeto(outname)
      logger.info('%s-imag beam written to %s' % (p[0], outname))
      outnames.append(outname)
    return outnames

  # see Greisen & Calabretta 2002, 395, 1061
  # Table 7
  # I=1
  # XX=-5
  # YY=-6
  # XY=-7
  # YX=-8
  if freqfirst:
    f[ext].header['CRVAL4'] = -5.0
  else:
    f[ext].header['CRVAL3'] = -5.0
  root = os.path.splitext(filename)[0]
  outname = root + '_beamXX' + '.fits'
  outnames = [outname]
  if os.path.exists(outname):
    os.remove(outname)
  f[ext].writeto(outname)
  logger.info('XX beam written to %s' % outname)

  if freqfirst:
    f[ext].data[0, :] = tempY[0, :]
  else:
    f[ext].data[:, 0] = tempY[:, 0]
  if freqfirst:
    f[ext].header['CRVAL4'] = -6.0
  else:
    f[ext].header['CRVAL3'] = -6.0
  outname = root + '_beamYY' + '.fits'
  if os.path.exists(outname):
    os.remove(outname)
  f[ext].writeto(outname)
  outnames.append(outname)
  logger.info('YY beam written to %s' % outname)

  logger.debug('Time (finshed): %s' % datetime.datetime.now().time())
  return outnames
