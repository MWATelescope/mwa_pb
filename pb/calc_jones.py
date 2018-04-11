"""Calculation of jones matrix with 2016, 2014 and analytic models - for testing and comparison"""

# DEBUGGER:
import pdb

import datetime
import logging
import math

from optparse import OptionParser, OptionGroup

try:
  import astropy.io.fits as pyfits
  import astropy.wcs as pywcs
  _useastropy = True
except ImportError:
  import pywcs
  import pyfits
  _useastropy = False


import numpy as np

from mwapy import ephem_utils
import mwa_sweet_spots


# configure the logging
logging.basicConfig(format='# %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)
# logger.setLevel(logging.WARNING)
logger.setLevel(logging.DEBUG)


import pb
import primary_beam
import beam_tools
h5filepath = pb.h5file


######################################################################
def get_azza_from_fits(filename, ext=0, precess=True):
  """
Get frequency & az & ZA arrays from fits file
  """
  logger.info('Time (start): %s' % datetime.datetime.now().time())

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
  f.close()

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
    logger.info('axis 3 is FREQ, axis 4 is STOKES')
    nfreq = h['NAXIS3']  # read number of frequency channels
    df = h['CDELT3']  # read frequency increment
  else:
    logger.info('axis 3 is STOKES, axis 4 is FREQ')
    nfreq = h['NAXIS4']
    df = h['CDELT4']
  logger.info('Number of frequency channels = ' + str(nfreq))
  # construct the basic arrays
  x = np.arange(1, h['NAXIS1'] + 1)
  y = np.arange(1, h['NAXIS2'] + 1)
  # assume we want the first frequency
  # if we have a cube this will have to change
  ff = 1
  # X,Y=np.meshgrid(x,y)
  Y, X = np.meshgrid(y, x)

  Xflat = X.flatten()
  Yflat = Y.flatten()
  FF = ff * np.ones(Xflat.shape)
  Tostack = [Xflat, Yflat, FF]
  for i in xrange(3, naxes):
    Tostack.append(np.ones(Xflat.shape))
  pixcrd = np.vstack(Tostack).transpose()

  try:
    # Convert pixel coordinates to world coordinates
    # The second argument is "origin" -- in this case we're declaring we
    # have 1-based (Fortran-like) coordinates.
    if _useastropy:
      sky = wcs.wcs_pix2world(pixcrd, 1)
    else:
      sky = wcs.wcs_pix2sky(pixcrd, 1)
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
  freq = freq[np.isfinite(freq)][0]
  if nfreq > 1:
    frequencies = np.arange(nfreq) * df + freq
  else:
    frequencies = np.array([freq])

  # and make them back into arrays
  RA = ra.reshape(X.shape)
  Dec = dec.reshape(Y.shape)

  # get the date so we can convert to Az,El
  try:
    d = h['DATE-OBS']
  except:
    logger.error('Unable to read observation date DATE-OBS from %s' % filename)
    return None
  if '.' in d:
    d = d.split('.')[0]
  dt = datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
  mwatime = ephem_utils.MWATime(datetime=dt)
  logger.info('Computing for %s' % mwatime)

  if precess:
    RAnow, Decnow = ephem_utils.precess(RA, Dec, 2000, mwatime.epoch)
  else:
    RAnow, Decnow = RA, Dec

  HA = float(mwatime.LST) - RAnow
  mwa = ephem_utils.Obs[ephem_utils.obscode['MWA']]
  Az, Alt = ephem_utils.eq2horz(HA, Decnow, mwa.lat)
  # go from altitude to zenith angle
  theta = (90 - Alt) * math.pi / 180
  phi = Az * math.pi / 180

  logger.info('Time (get beam): %s' % datetime.datetime.now().time())

  return {'za_rad': theta, 'astro_az_rad': phi, 'frequencies': frequencies}


def get_IQUV(filename):
  """Get IQUV from a CASA im file exported to fits."""
  f = pyfits.open(filename)
  data = f[0].data
  f.close()
  stokes = {}
  stokes['I'] = data[0, 0, :, :]
  stokes['Q'] = data[1, 0, :, :]
  stokes['U'] = data[2, 0, :, :]
  stokes['V'] = data[3, 0, :, :]
  return stokes


def get_inst_pols(stokes):
  """Return instrumental polaristations matrix (Vij)"""
  XX = stokes['I'] + stokes['Q']
  XY = stokes['U'] + stokes['V'] * 1j
  YX = stokes['U'] - stokes['V'] * 1j
  YY = stokes['I'] - stokes['Q']

  Vij = np.array([[XX, XY], [YX, YY]])

  return Vij


# def Vij_beam_correct(j,Vij,centre=None):
#    """Corrects Vij for the beam amplitude. 
#    This is required when beam correction has not been done during calibration.
#    Assumes identical beam patterns.
#    Assumes calibrator source is at centre of image"""
#    
#    if centre==None:
#        my_shape=Vij[0,0,:,:].shape
#        centre=(my_shape[0]/2,my_shape[1]/2) #Cal source at image centre
#        logger.warning('Using centre of image as calibrator location')
#    
#    correction=beam_tools.makeUnpolInstrumentalResponse(j[:,:,centre[0],centre[1]],j[:,:,centre[0],centre[1]])    
##    XX=temp[0,0]
##    YY=temp[1,1]
##    XY=temp[0,1]
##    YX=temp[1,0]
##  
##    correction=np.array([[XX,XX**0.5*YY**0.5],[XX**0.5*YY**0.5,YY]])
##    correction=np.array([[XX,XY],[YX,YY]])
##    correction=np.array([[XX,1],[1,YY]])    
#    logger.warning('Calibration correction factors: XX=%s, XY=%s, YX=%s, YY=%s'%
#    (correction[0,0], correction[0,1], correction[1,0], correction[1,1]))
#    #Tile 2x2 correction matrix apply to Vij
##    Vij_corrected=Vij*np.tile(correction[:, :, np.newaxis, np.newaxis],(my_shape[0],my_shape[1]))
#    Vij_corrected=Vij*0.0
##    UnpolInstrumentalResponse=makeUnpolInstrumentalResponse(j[:,:,centre[0],centre[1]],j[:,:,centre[0],centre[1]])
#    
#    for i in [0,1]:
#        for ii in [0,1]:    
#                Vij_corrected[i,ii]=correction[i,ii]*Vij[i,ii]
#    return Vij_corrected
#    


def Dij_beam_correct(j, Dij, centre=None):
  """Corrects Dij:
      Dij'' = E^-1 Dij' (E^H)^-1

  This is required when beam correction has not been done during calibration.
  Assumes identical beam patterns.
  Uses j at direction 'centre' """

  my_shape = Dij.shape
  if len(my_shape) == 4:
    if centre is None:
      centre = (Dij[0, 0, :, :].shape[0] / 2, Dij[0, 0, :, :].shape[1] / 2)  # Cal source at image centre
      logger.warning('Using centre of image as calibrator location')

    # Get jones at location of calibrator source
    my_j = j[:, :, centre[0], centre[1]]
    logger.debug('My Jones correction:\n[%s %s]\n[%s %s]'
                 % (my_j[0, 0], my_j[0, 1], my_j[1, 0], my_j[1, 1]))

    temp = beam_tools.makeUnpolInstrumentalResponse(my_j, my_j)
    logger.debug('My Jones correction:\n[XX:%s XY:%s]\n[YX:%s YY:%s]'
                 % (temp[0, 0], temp[0, 1], temp[1, 0], temp[1, 1]))

    out = Dij * 0  # Copy complex array

    # my_jinv = inv2x2(my_j)  # J1^-1

    my_jH = np.transpose(my_j.conj())  # J2^H
    # my_jHinv = inv2x2(my_jH)  # (J2^H)^-1

    for i in range(my_shape[2]):
      # Matrix muliply every 2x2 array in the 2-D sky
      if i % 1000 == 0:
        logger.debug('Processing %s of %s...' % (i, my_shape[2]))
      # print 'Processing  %s of %s...'%(i,my_shape[2])
      for j in range(my_shape[3]):
        # J1^-1 (Vij (J2^H)^-1)
        # out[:,:,i,j]=np.dot(my_jinv, np.dot(Dij[:,:,i,j], my_jHinv))
        out[:, :, i, j] = np.dot(my_j, np.dot(Dij[:, :, i, j], my_jH))
  else:
    print 'FIXME for other lengths!!'
    return
  return out


def Vij_beam_correct(j, Vij, centre=None):
  """Corrects Vij for the beam amplitude.
  This is required when beam correction has not been done during calibration.
  Assumes identical beam patterns.
  Assumes calibrator source is at centre of image"""

  my_shape = Vij[0, 0, :, :].shape
  if centre is None:
    centre = (my_shape[0] / 2, my_shape[1] / 2)  # Cal source at image centre
    logger.warning('Using centre of image as calibrator location')

  temp = beam_tools.makeUnpolInstrumentalResponse(j[:, :, centre[0], centre[1]], j[:, :, centre[0], centre[1]])
  XX = temp[0, 0]
  YY = temp[1, 1]
  # XY = temp[0, 1]
  # YX = temp[1, 0]

  correction = np.array([[XX, XX ** 0.5 * YY ** 0.5], [XX ** 0.5 * YY ** 0.5, YY]])
  # correction=np.array([[XX,XY],[YX,YY]])
  # correction=np.array([[XX,1],[1,YY]])
  logger.warning('Calibration correction factors: XX=%s, XY=%s, YX=%s, YY=%s' %
                 (correction[0, 0], correction[0, 1], correction[1, 0], correction[1, 1]))
  # Tile 2x2 correction matrix apply to Vij
  Vij_corrected = Vij * np.tile(correction[:, :, np.newaxis, np.newaxis], (my_shape[0], my_shape[1]))

  return Vij_corrected


def inv2x2(j):
  """Invert 2x2 matrix of any number of additional dimensions:
  A^-1=[[a b][c d]]^-1 = 1/(ad-bc) * [[d -b][-c a]]"""
  return 1 / (j[0, 0] * j[1, 1] - j[0, 1] * j[1, 0]) * np.array([[j[1, 1], -1 * j[0, 1]], [-1 * j[1, 0], j[0, 0]]])


def estimateSkyBrightnessMatrix(j1, j2, Vij):
  """B = E^-1 Bapp (E^H)^-1"""
  my_shape = Vij.shape
  b = Vij * 0  # Copy complex array

  j1inv = inv2x2(j1)  # J1^-1

  if len(my_shape) == 4:
    j2H = np.transpose(j2.conj(), (1, 0, 2, 3))  # J2^H
    j2Hinv = inv2x2(j2H)  # (J2^H)^-1

    for i in range(my_shape[2]):
      # Matrix muliply every 2x2 array in the 2-D sky
      if i % 1000 == 0:
        logger.debug('Processing %s of %s...' % (i, my_shape[2]))
      #                print 'Processing  %s of %s...'%(i,my_shape[2])
      for j in range(my_shape[3]):
        # J1^-1 (Vij (J2^H)^-1)
        b[:, :, i, j] = np.dot(j1inv[:, :, i, j], np.dot(Vij[:, :, i, j], j2Hinv[:, :, i, j]))
  else:
    print 'FIXME for other lengths!!'
    return
  return b


# def estimateSkyBrightnessMatrix(j1,j2,Vij):
#    my_shape=Vij.shape
#    b=Vij*0 #Copy complex array
#
#    if len(my_shape)==4:
#        for i in range(my_shape[2]):
#            if i%100==0:
#                logger.debug('Processing %s of %s...'%(i,my_shape[2]))
#                print 'Processing  %s of %s...'%(i,my_shape[2])
#            for j in range(my_shape[3]):  
#                j1inv=np.linalg.inv(j1[:,:,i,j]) #J1^-1               
#                j2H=j2[:,:,i,j].conj().T #J2^H
#                j2Hinv=np.linalg.inv(j2H) #(J2^H)^-1
#                
#                #J1^-1 (Vij (J2^H)^-1)
#                b[:,:,i,j]=np.dot(j1inv, np.dot(Vij[:,:,i,j], j2Hinv))
#                
#    else:
#        print 'FIXME for other lengths!!'
#        return    
#    return b


def B2IQUV(B):
  """Convert sky brightness matrix to I, Q, U, V"""
  B11 = B[0, 0, :, :]
  B12 = B[0, 1, :, :]
  B21 = B[1, 0, :, :]
  B22 = B[1, 1, :, :]

  stokes = {}
  stokes['I'] = (B11 + B22) / 2.
  stokes['Q'] = (B11 - B22) / 2.
  stokes['U'] = (B12 + B21) / 2.
  stokes['V'] = 1j * (B21 - B12) / 2.
  return stokes


def calc_ratio(dec, target_freq_Hz, gridpoint, h5filepath):
  za = dec + 26.7
  az = 0

  print "\n\n\n\n"
  print "################################# DEC = %.2f #################################" % dec
  za_rad = za * math.pi / 180.0
  az_rad = az * math.pi / 180.0
  za_rad = np.array([[za_rad]])
  az_rad = np.array([[az_rad]])

  # non-realistic scenario - I am ~tracking the source with the closest sweetspot - whereas I should be staying at the same sweetspot
  #   gridpoint=find_closest_gridpoint(za)
  #
  print "dec=%.4f [deg] -> za=%.4f [deg] - %s" % (dec, za, gridpoint)
  delays = gridpoint[4]
  delays = np.vstack((delays, delays))
  print delays
  print "-----------"

  Jones_FullEE = primary_beam.MWA_Tile_full_EE(za_rad, az_rad, target_freq_Hz, delays=delays, zenithnorm=True,
                                               jones=True, interp=True)
  # swap axis to have Jones martix in the 1st
  Jones_FullEE_swap = np.swapaxes(np.swapaxes(Jones_FullEE, 0, 2), 1, 3)

  # TEST if equivalent to :
  Jones_FullEE_2D = Jones_FullEE_swap[:, :, 0, 0]
  #   Jones_FullEE_2D=np.array([ [Jones_FullEE_swap[0,0][0][0],Jones_FullEE_swap[0,1][0][0]] , [Jones_FullEE_swap[1,0][0][0],Jones_FullEE_swap[1,1][0][0]] ])
  print "Jones FullEE:"
  print "----------------------"
  print Jones_FullEE
  print "----------------------"
  print Jones_FullEE_2D
  print "----------------------"

  #  Average Embeded Element model:
  print "size(delays) = %d" % np.size(delays)
  Jones_AEE = primary_beam.MWA_Tile_advanced(za_rad, az_rad, target_freq_Hz, delays=delays, zenithnorm=True, jones=True)
  #   Jones_AEE=primary_beam.MWA_Tile_advanced( np.array([[0]]), np.array([[0]]), target_freq_Hz,delays=delays, zenithnorm=True, jones=True)
  Jones_AEE_swap = np.swapaxes(np.swapaxes(Jones_AEE, 0, 2), 1, 3)
  Jones_AEE_2D = Jones_AEE_swap[:, :, 0, 0]
  #   Jones_AEE_2D=np.array([ [Jones_AEE_swap[0,0][0][0],Jones_AEE_swap[0,1][0][0]] , [Jones_AEE_swap[1,0][0][0],Jones_AEE_swap[1,1][0][0]] ])
  print "----------------------"
  print "Jones AEE:"
  print "----------------------"
  print Jones_AEE
  print "----------------------"
  print Jones_AEE_2D
  print "----------------------"

  # Analytical Model:
  #   beams={}
  #   beams['XX'],beams['YY']=primary_beam.MWA_Tile_analytic(za_rad, az_rad, target_freq_Hz, delays=delays, zenithnorm=True, jones=True)
  Jones_Anal = primary_beam.MWA_Tile_analytic(za_rad, az_rad, target_freq_Hz, delays=delays, zenithnorm=True,
                                              jones=True)
  Jones_Anal_swap = np.swapaxes(np.swapaxes(Jones_Anal, 0, 2), 1, 3)
  Jones_Anal_2D = Jones_Anal_swap[:, :, 0, 0]
  #   Jones_Anal_2D=np.array([ [Jones_Anal_swap[0,0][0][0],Jones_Anal_swap[0,1][0][0]] , [Jones_Anal_swap[1,0][0][0],Jones_Anal_swap[1,1][0][0]] ])
  print "----------------------"
  print "Jones Analytic:"
  print "----------------------"
  print Jones_Anal
  print "----------------------"
  print Jones_Anal_2D
  print "----------------------"
  print "TEST:"
  print "----------------------"
  print "%.8f    %.8f" % (Jones_Anal_2D[0, 0], Jones_Anal_2D[0, 1])
  print "%.8f    %.8f" % (Jones_Anal_2D[1, 0], Jones_Anal_2D[1, 1])
  print "----------------------"

  # Use Jones_FullEE_2D as REAL sky and then ...
  B_sky = np.array([[1, 0], [0, 1]])
  Jones_FullEE_2D_H = np.transpose(Jones_FullEE_2D.conj())
  B_app = np.dot(Jones_FullEE_2D, np.dot(B_sky, Jones_FullEE_2D_H))  # E x B x E^H

  # test the procedure itself:
  Jones_FullEE_2D_H_Inv = inv2x2(Jones_FullEE_2D_H)
  Jones_FullEE_2D_Inv = inv2x2(Jones_FullEE_2D)
  B_sky_cal = np.dot(Jones_FullEE_2D_Inv, np.dot(B_app, Jones_FullEE_2D_H_Inv))
  print B_sky
  print "Recovered using FullEE model:"
  print B_sky_cal

  # calibrate back using AEE model :
  #   Jones_AEE_2D=Jones_Anal_2D # overwrite Jones_AEE_2D with Analytic to use it for calibration
  Jones_AEE_2D_H = np.transpose(Jones_AEE_2D.conj())
  Jones_AEE_2D_H_Inv = inv2x2(Jones_AEE_2D_H)
  Jones_AEE_2D_Inv = inv2x2(Jones_AEE_2D)
  B_sky_cal = np.dot(Jones_AEE_2D_Inv, np.dot(B_app, Jones_AEE_2D_H_Inv))
  print "Recovered using AEE model:"
  print B_sky_cal
  # I_cal = B_sky_cal[0, 0] + B_sky_cal[1, 1]
  #   print "FINAL : %.8f ratio = %.8f / 2 = %.8f" % (dec,abs(I_cal),(abs(I_cal)/2.00))
  # ratio = abs(B_sky_cal[0][0] / B_sky_cal[1][1])
  # FINAL VALUES for the paper to compare with Figure 4 in GLEAM paper :
  ratio_ms = B_sky_cal[0][0] / B_sky_cal[1][1]
  gleam_ratio = ratio_ms
  gleam_XX = B_sky_cal[0][0]
  gleam_YY = B_sky_cal[1][1]
  # gleam_q_leakage = (B_sky_cal[0][0] - B_sky_cal[1][1]) / (B_sky_cal[0][0] + B_sky_cal[1][1])
  print "DEBUG (DEC = %.2f deg) : GLEAM-ratio = %.4f = (%s / %s)" % (dec, gleam_ratio, gleam_XX, gleam_YY)

  return (gleam_ratio.real, gleam_XX, gleam_YY)


def parse_options():
  usage = "Usage: %prog [options]\n"
  usage += '\tTest of calibrating true sky (represented by FEE model) by AEE model\n'
  parser = OptionParser(usage=usage, version=1.00)
  parser.add_option('-a', '--az', dest="az", default="45", help="Azimuth [default %default deg]", metavar="FLOAT",
                    type="float")
  parser.add_option('-z', '--za', dest="za", default="19.6", help="Zenith distance [default %default deg]",
                    metavar="FLOAT", type="float")
  parser.add_option('-f', '--freq', dest="freq", default="205", help="Freq in MHz [default %default MHz]",
                    metavar="FLOAT", type="float")
  (options, args) = parser.parse_args()

  return (options, args)


def calc_jones(az, za, target_freq_Hz=205e6):
  za_rad = za * math.pi / 180.0
  az_rad = az * math.pi / 180.0
  za_rad = np.array([[za_rad]])
  az_rad = np.array([[az_rad]])

  gridpoint = mwa_sweet_spots.find_closest_gridpoint(az, za)
  print "Az = %.2f [deg], za = %.2f [deg], gridpoint = %d" % (az, za, gridpoint[0])

  delays = gridpoint[4]
  delays = np.vstack((delays, delays))
  print delays
  print "-----------"

  Jones_FullEE = primary_beam.MWA_Tile_full_EE(za_rad, az_rad, target_freq_Hz, delays=delays, zenithnorm=True,
                                               jones=True, interp=True)
  # swap axis to have Jones martix in the 1st
  Jones_FullEE_swap = np.swapaxes(np.swapaxes(Jones_FullEE, 0, 2), 1, 3)

  # TEST if equivalent to :
  Jones_FullEE_2D = Jones_FullEE_swap[:, :, 0, 0]
  #   Jones_FullEE_2D=np.array([ [Jones_FullEE_swap[0,0][0][0],Jones_FullEE_swap[0,1][0][0]] , [Jones_FullEE_swap[1,0][0][0],Jones_FullEE_swap[1,1][0][0]] ])
  print "Jones FullEE:"
  print "----------------------"
  print Jones_FullEE
  print "----------------------"
  print Jones_FullEE_2D
  print "----------------------"

  beams = {}
  beams['XX'], beams['YY'] = primary_beam.MWA_Tile_full_EE(za_rad, az_rad, target_freq_Hz, delays=delays,
                                                           zenithnorm=True, power=True)
  print "Beams power = %.4f / %.4f" % (beams['XX'], beams['YY'])

  # Average Embeded Element model:
  print "size(delays) = %d" % np.size(delays)
  Jones_AEE = primary_beam.MWA_Tile_advanced(za_rad, az_rad, target_freq_Hz, delays=delays, zenithnorm=True, jones=True)
  #   Jones_AEE=primary_beam.MWA_Tile_advanced( np.array([[0]]), np.array([[0]]), target_freq_Hz,delays=delays, zenithnorm=True, jones=True)
  Jones_AEE_swap = np.swapaxes(np.swapaxes(Jones_AEE, 0, 2), 1, 3)
  Jones_AEE_2D = Jones_AEE_swap[:, :, 0, 0]
  #   Jones_AEE_2D=np.array([ [Jones_AEE_swap[0,0][0][0],Jones_AEE_swap[0,1][0][0]] , [Jones_AEE_swap[1,0][0][0],Jones_AEE_swap[1,1][0][0]] ])
  print "----------------------"
  print "Jones AEE:"
  print "----------------------"
  print Jones_AEE
  print "----------------------"
  print Jones_AEE_2D
  print "----------------------"

  Jones_Anal = primary_beam.MWA_Tile_analytic(za_rad, az_rad, target_freq_Hz, delays=delays, zenithnorm=True,
                                              jones=True)
  Jones_Anal_swap = np.swapaxes(np.swapaxes(Jones_Anal, 0, 2), 1, 3)
  Jones_Anal_2D = Jones_Anal_swap[:, :, 0, 0]

  print "----------------------  COMPARISON OF JONES MATRICES FEE vs. AEE ----------------------"
  print "Jones FullEE:"
  print "----------------------"
  print Jones_FullEE_2D
  print "----------------------"
  print
  print "Jones AEE:"
  print "----------------------"
  print Jones_AEE_2D
  print "----------------------"
  print
  print "----------------------"
  print "Jones Analytic:"
  print "----------------------"
  print Jones_Anal_2D
  print "----------------------"

  return (Jones_FullEE_2D, Jones_AEE_2D, Jones_Anal_2D)


if __name__ == "__main__":
  logger.setLevel(logging.DEBUG)

  # if len(sys.argv) > 2:
  #   outfile=sys.argv[2]

  (options, args) = parse_options()

  target_freq_Hz = options.freq * 1e6
  target_freq_MHz = options.freq
  outfile = "ratio_vs_dec_%.2fMHz.out" % (target_freq_MHz)

  print "-------------------------------------------"
  print "PARAMETERS:"
  print "-------------------------------------------"
  print "Freq    = %.2f [Hz]" % (target_freq_Hz)
  print "(az,za) = (%.2f,%.2f) [deg]" % (options.az, options.za)
  print "Outfile = %s" % (outfile)
  print "-------------------------------------------"

  outf = open(outfile, "w")
  outf.write("# DEC[deg]  S_xx/S_yy   Q_leakage GLEAM-RATIO-MS-way GLEAM-Ratio-DanielWAY\n")
  outf.close()

  az = options.az
  za = options.za

  #   if az < 0 :
  #      az = 360 + az

  calc_jones(az, za, target_freq_Hz=target_freq_Hz)
