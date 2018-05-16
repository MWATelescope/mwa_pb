"""
$Rev: 4142 $:     Revision of last commit
$Author: dkaplan $:  Author of last commit
$Date: 2011-10-31 11:30:40 -0500 (Mon, 31 Oct 2011) $:    Date of last commit

"""

import logging
import math

import numpy

import astropy
from astropy.time import Time
from astropy.coordinates import SkyCoord

import beam_full_EE
import config
import metadata
import mwa_tile

logging.basicConfig(format='# %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)  # default logger level is WARNING


#########
#########
def MWA_Tile_full_EE(za, az, freq,
                     delays=None,
                     zenithnorm=True,
                     power=True,
                     jones=False,
                     interp=True,
                     pixels_per_deg=5):
  """
  Use the new MWA tile model from beam_full_EE.py that includes mutual coupling
  and the simulated dipole response. Returns the XX and YY response to an
  unpolarised source.

  if jones=True, will return the Jones matrix instead of the XX,YY response.
  In this case, the power flag will be ignored.
  If interp=False, the pixels_per_deg will be ignored

  delays should be a numpy array of size (2,16), although a (16,) list or a (16,) array will also be accepted

  az - azimuth angles (radians), north through east.
  za - zenith angles (radian)
  """
  tile = beam_full_EE.ApertureArray(config.h5file, freq)
  mybeam = beam_full_EE.Beam(tile, delays, amps=numpy.ones([2, 16]))  # calling with amplitudes=1 every time - otherwise they get overwritten !!!
  if interp:
    j = mybeam.get_interp_response(az, za, pixels_per_deg)
  else:
    j = mybeam.get_response(az, za)
  if zenithnorm:
    j = tile.apply_zenith_norm_Jones(j)  # Normalise

  # TODO: do frequency interpolation here (with 2nd adjacent beam)

  # Use swapaxis to place jones matrices in last 2 dimensions
  # insead of first 2 dims.
  if len(j.shape) == 4:
    j = numpy.swapaxes(numpy.swapaxes(j, 0, 2), 1, 3)
  elif len(j.shape) == 3:  # 1-D
    j = numpy.swapaxes(numpy.swapaxes(j, 1, 2), 0, 1)
  else:  # single value
    pass

  if jones:
    return j

  # Use mwa_tile makeUnpolInstrumentalResponse because we have swapped axes
  vis = mwa_tile.makeUnpolInstrumentalResponse(j, j)
  if not power:
    return (numpy.sqrt(vis[:, :, 0, 0].real), numpy.sqrt(vis[:, :, 1, 1].real))
  else:
    return (vis[:, :, 0, 0].real, vis[:, :, 1, 1].real)


#########
#########
def MWA_Tile_advanced(za, az, freq=100.0e6, delays=None, power=True, jones=False):
  """
  Use the new MWA tile model from mwa_tile.py that includes mutual coupling
  and the simulated dipole response. Returns the XX and YY response to an
  unpolarised source.

  if jones=True, will return the Jones matrix instead

  delays should be a numpy array of size (2,16), although a (16,) list or a (16,) array will also be accepted

  """
  if isinstance(delays, list):
    delays = numpy.array(delays)

  if delays.shape == (16,):
    try:
      delays = numpy.repeat(numpy.reshape(delays, (1, 16)), 2, axis=0)
    except:
      logger.error('Unable to convert delays (shape=%s) to (2,16)' % (delays.shape))
      return None
  assert delays.shape == (2, 16), "Delays %s have unexpected shape %s" % (delays, delays.shape)

  d = mwa_tile.Dipole(atype='lookup')

  logger.debug("Delays: " + str(delays))
  tile = mwa_tile.ApertureArray(dipoles=[d] * 16)  # tile of identical dipoles
  j = tile.getResponse(az, za, freq, delays=delays)
  if jones:
    return j
  vis = mwa_tile.makeUnpolInstrumentalResponse(j, j)
  if not power:
    return (numpy.sqrt(vis[:, :, 0, 0].real), numpy.sqrt(vis[:, :, 1, 1].real))
  else:
    return (vis[:, :, 0, 0].real, vis[:, :, 1, 1].real)


######################################################################
# Based on code from Daniel Mitchel
# 2012-02-13
# taken from the RTS codebase
######################################################################
def MWA_Tile_analytic(za, az,
                      freq=100.0e6,
                      delays=None,
                      zenithnorm=True,
                      power=False,
                      dipheight=config.DIPOLE_HEIGHT,
                      dip_sep=config.DIPOLE_SEPARATION,
                      delay_int=config.DELAY_INT,
                      jones=False,
                      amps=None):
  """
  gainXX,gainYY=MWA_Tile_analytic(za, az, freq=100.0e6, delays=None, zenithnorm=True, power=True, dipheight=0.278, dip_sep=1.1, delay_int=435.0e-12)
  if power=False, then gains are voltage gains - should be squared for power
  otherwise are power

  za is zenith-angle in radians
  az is azimuth in radians, phi=0 points north
  freq in Hz, height, sep in m

  delays should be a numpy array of size (2,16), although a (16,) list or a (16,) array will also be accepted

  """
  theta = za
  phi = az

  c = 2.998e8
  # wavelength in meters
  lam = c / freq

  if (delays is None):
    delays = 0

  if (isinstance(delays, float) or isinstance(delays, int)):
    delays = delays * numpy.ones((16))
  if (isinstance(delays, numpy.ndarray) and len(delays) == 1):
    delays = delays[0] * numpy.ones((16))
  if isinstance(delays, list):
    delays = numpy.array(delays)

  assert delays.shape == (2, 16) or delays.shape == (16,), "Delays %s have unexpected shape %s" % (delays, delays.shape)
  if len(delays.shape) > 1:
    delays = delays[0]

  if amps is None:
    amps = numpy.ones((16))

  # direction cosines (relative to zenith) for direction az,za
  projection_east = numpy.sin(theta) * numpy.sin(phi)
  projection_north = numpy.sin(theta) * numpy.cos(phi)
  projection_z = numpy.cos(theta)

  # dipole position within the tile
  dipole_north = dip_sep * numpy.array([1.5, 1.5, 1.5, 1.5,
                                        0.5, 0.5, 0.5, 0.5, -
                                        0.5, -0.5, -0.5, -0.5,
                                        -1.5, -1.5, -1.5, -1.5])
  dipole_east = dip_sep * numpy.array([-1.5, -0.5, 0.5, 1.5,
                                       -1.5, -0.5, 0.5, 1.5,
                                       -1.5, -0.5, 0.5, 1.5,
                                       -1.5, -0.5, 0.5, 1.5])
  dipole_z = dip_sep * numpy.zeros(dipole_north.shape)

  # loop over dipoles
  array_factor = 0.0

  non_zero_count = 0
  for i in xrange(4):
    for j in xrange(4):
      k = 4 * j + i
      # relative dipole phase for a source at (theta,phi)
      phase = amps[k] * numpy.exp((1j) * 2 * math.pi / lam * (dipole_east[k] * projection_east +
                                                              dipole_north[k] * projection_north +
                                                              dipole_z[k] * projection_z -
                                                              delays[k] * c * delay_int))
      if amps[k] != 0:
        non_zero_count = non_zero_count + 1
      array_factor += phase / 16.0
      # array_factor+=phase

  # print "non_zero_count = %d" % non_zero_count
  # array_factor = array_factor / non_zero_count

  ground_plane = 2 * numpy.sin(2 * math.pi * dipheight / lam * numpy.cos(theta))
  # make sure we filter out the bottom hemisphere
  ground_plane *= (theta <= math.pi / 2)
  # normalize to zenith
  if (zenithnorm):
    # print "Normalisation factor (analytic) = %.4f" % (2*numpy.sin(2*math.pi*dipheight/lam))
    ground_plane /= 2 * numpy.sin(2 * math.pi * dipheight / lam)

  # response of the 2 tile polarizations
  # gains due to forshortening
  dipole_ns = numpy.sqrt(1 - projection_north * projection_north)
  dipole_ew = numpy.sqrt(1 - projection_east * projection_east)

  # voltage responses of the polarizations from an unpolarized source
  # this is effectively the YY voltage gain
  gain_ns = dipole_ns * ground_plane * array_factor
  # this is effectively the XX voltage gain
  gain_ew = dipole_ew * ground_plane * array_factor

  if jones:
    # Calculate Jones matrices
    dipole_jones = numpy.array([[numpy.cos(theta) * numpy.sin(phi), 1 * numpy.cos(phi)],
                                [numpy.cos(theta) * numpy.cos(phi), -numpy.sin(phi)]])
    j = dipole_jones * ground_plane * array_factor
    # print "dipole_jones = %s" % (dipole_jones)
    # print "ground_plane = %s , array_factor = %s" % (ground_plane,array_factor)

    # Use swapaxis to place jones matrices in last 2 dimensions
    # insead of first 2 dims.
    if len(j.shape) == 4:
      j = numpy.swapaxes(numpy.swapaxes(j, 0, 2), 1, 3)
    elif len(j.shape) == 3:  # 1-D
      j = numpy.swapaxes(numpy.swapaxes(j, 1, 2), 0, 1)
    else:  # single value
      pass
    return j

  if power:
    return numpy.real(numpy.conj(gain_ew) * gain_ew), numpy.real(numpy.conj(gain_ns) * gain_ns)
  return gain_ew, gain_ns


def analytic_full_EE_correction(za, az, freq, delays):
  """Calculate correction matrix for antenna-based complex gains (gx, gy)
  to go from analytic to full_EE models

  Input:
  a single za, az direction in radians

  Output:
  correction_matrix - [[gx, 0], [0, gy]], where we expect the cross-terms to be 0"""

  # Analytic Jones matrix
  za = numpy.array(za)
  az = numpy.array(az)

  j_ana = MWA_Tile_analytic(za, az, freq, delays, jones=True, zenithnorm=False)

  # Spherical harmonics Jones matrix
  j_full_EE = MWA_Tile_full_EE(za, az, freq,
                               delays=delays,
                               zenithnorm=False,
                               jones=True,
                               # model_ver=model_ver,
                               interp=False)

  # .conj().T -> hermitian
  if len(za.shape) == 0:
    j_full_EE = j_full_EE[:, :, 0]  # Drop last axis

    correction_matrix = numpy.dot(numpy.dot(numpy.linalg.inv(j_ana),
                                            numpy.dot(j_full_EE,
                                                      j_full_EE.conj().T)),
                                  numpy.linalg.inv(j_ana.conj().T))
  else:
    e = 'ZA, Az arrays are not currently supported in analytic_full_EE_correction'
    # logger.error(e)
    raise ValueError(e)

  return correction_matrix


##################################################
def get_beam_response(obsid,
                      sources,
                      dt=296,
                      centeronly=True):
  """
  Power=get_beam_response(obsid,sources, dt=296,
  centeronly=True,
  verbose=False)

  sources=[[RA,Dec],
  [RA,Dec],
  ...
  ]

  returns observation_metadata, times, ResponseX, ResponseY
  both X and Y responses are [#sources, #times, #frequencies]

  Example:
  sources=[[83.633075, +22.0144944444444]]

  obsid=1099415632
  observation,times,RX,RY=get_beam_response(obsid,sources,centeronly=False)


  """
  observation = metadata.get_observation(obsid=obsid)
  if observation is None:
    logger.error('Unable to retrieve metadata for observation %d' % obsid)
    return None

  duration = observation['starttime'] - observation['stoptime']
  starttimes = numpy.arange(0, duration, dt)
  stoptimes = starttimes + dt
  stoptimes[stoptimes > duration] = duration
  Ntimes = len(starttimes)
  midtimes = obsid + 0.5 * (starttimes + stoptimes)
  logger.info('Will output for %d times from 0 to %ds after %d\n' % (Ntimes, duration, obsid))

  channels = observation['rfstreams']['0']['frequencies']
  if not centeronly:
    PowersX = numpy.zeros((len(sources),
                           Ntimes,
                           len(channels)))
    PowersY = numpy.zeros((len(sources),
                           Ntimes,
                           len(channels)))
    # in Hz
    frequencies = numpy.array(channels) * 1.28e6
  else:
    PowersX = numpy.zeros((len(sources),
                           Ntimes, 1))
    PowersY = numpy.zeros((len(sources),
                           Ntimes, 1))
    frequencies = numpy.array([channels[12]]) * 1.28e6   # center channel
  RAs = numpy.array([x[0] for x in sources])
  Decs = numpy.array([x[1] for x in sources])
  if len(RAs) == 0:
    logger.error('Must supply >=1 source positions\n')
    return None
  if not len(RAs) == len(Decs):
    logger.error('Must supply equal numbers of RAs and Decs\n')
    return None

  obs_source = SkyCoord(ra=RAs,
                        dec=Decs,
                        equinox='J2000',
                        unit=(astropy.units.deg, astropy.units.deg))
  obs_source.location = config.MWAPOS

  for itime in xrange(Ntimes):
    obs_source.obstime = Time(midtimes[itime], format='gps', scale='utc')
    obs_source_prec = obs_source.transform_to('altaz')
    Azs, Alts = obs_source_prec.az.deg, obs_source_prec.alt.deg

    # go from altitude to zenith angle
    theta = numpy.radians(90 - Alts)
    phi = numpy.radians(Azs)

    for ifreq in xrange(len(frequencies)):
      rX, rY = MWA_Tile_analytic(theta, phi,
                                 freq=frequencies[ifreq], delays=observation['rfstreams']['0']['delays'],
                                 zenithnorm=True,
                                 power=True)
      PowersX[:, itime, ifreq] = rX
      PowersY[:, itime, ifreq] = rY

  return observation, midtimes, PowersX, PowersY
