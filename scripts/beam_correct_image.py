#!/usr/bin/env python

"""Tim's attempt to correct for Stokes leakage on images generated without 
beam correction durign calibration"""

# DEBUG :
# import pdb

import datetime
import logging
import math
from optparse import OptionParser
import sys

import astropy
from astropy.coordinates import SkyCoord, FK5
import astropy.io.fits as pyfits
import astropy.wcs as pywcs
from astropy.time import Time

import numpy as np

from mwa_pb import config
from mwa_pb import mwa_sweet_spots
from mwa_pb import primary_beam
from mwa_pb import beam_tools

# configure the logging
logging.basicConfig(format='# %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


######################################################################
def get_azza_from_fits(filename, ext=0, precess=True, freq_mhz=0, gps=0):
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
    nfreq = 1
    df = 0
    # try  order  RA,Dec,Freq,Stokes
    if 'RA' not in h['CTYPE1']:
        logger.error('Coordinate 1 should be RA')
        return None
    if 'DEC' not in h['CTYPE2']:
        logger.error('Coordinate 1 should be DEC')
        return None
    if naxes >= 4:
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
    Y, X = np.meshgrid(y, x)

    Xflat = X.flatten()
    Yflat = Y.flatten()
    FF = ff * np.ones(Xflat.shape)

    if naxes >= 4:
        Tostack = [Xflat, Yflat, FF]
        for i in xrange(3, naxes):
            Tostack.append(np.ones(Xflat.shape))
    else:
        Tostack = [Xflat, Yflat]
    pixcrd = np.vstack(Tostack).transpose()

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
    if naxes >= 4:
        if freqfirst:
            freq = sky[:, 2]
        else:
            freq = sky[:, 3]
        freq = freq[np.isfinite(freq)][0]
    else:
        freq = freq_mhz * 1000000

    if nfreq > 1:
        frequencies = np.arange(nfreq) * df + freq
    else:
        frequencies = np.array([freq])

    print "Frequency[0] = %.2f" % (frequencies[0])

    # and make them back into arrays
    RA = ra.reshape(X.shape)
    Dec = dec.reshape(Y.shape)

    # get the date so we can convert to Az,El
    try:
        d = h['DATE-OBS']
    except Exception:
        logger.error('Unable to read observation date DATE-OBS from %s' % filename)

        if gps > 0:
            time = Time(gps, format='gps', scale='utc')
            d = time.fits
            print "gps=%d -> d=%s" % (gps, d)
        else:
            logger.error('GPS time not provided either -> cannot continue')
            return None

    if '.' in d:
        d = d.split('.')[0]
    dt = datetime.datetime.strptime(d, '%Y-%m-%dT%H:%M:%S')
    mwatime = Time(dt)
    logger.info('Computing for %s=%10.0f' % (mwatime.iso, mwatime.gps))

    source = SkyCoord(ra=RA, dec=Dec, frame='icrs', unit=(astropy.units.deg, astropy.units.deg))
    source.location = config.MWAPOS
    source.obstime = mwatime

    source_altaz = source.transform_to('altaz')
    Alt, Az = source_altaz.alt.deg, source_altaz.az.deg  # Transform to Topocentric Alt/Az at the current epoch

    if precess:
        source_now = source.transform_to(FK5(equinox=mwatime))  # Transform to FK5 coordinates at the current epoch
        RAnow, Decnow = source_now.ra.deg, source_now.dec.deg
    else:
        RAnow, Decnow = RA, Dec

    # print "DEBUG = %s" % (Az) # different than what it was in the original version 
    # print "DEBUG = %s" % (Alt)

    # go from altitude to zenith angle
    theta = (90 - Alt) * math.pi / 180
    phi = Az * math.pi / 180

    logger.info('Time (get beam): %s' % datetime.datetime.now().time())

    # BUGFIX by MS 2016-04-20 - described in details in odt document. The essence is that python reads images transposed with respect to cross-diagonal
    #  however later Tim calculates map of ra,dec then az,alt using WCS parameters which results in having coordinates for non-transposed image.
    # I verified pixel values vs. RA,DEC in the map in ds9 and the code and calling .transpose() here fixes the problem.
    # Also makes images look correct, otherwise there were "leakage-sources" on the right (east) of the beam-corrected image !
    #    return {'za_rad':theta,'astro_az_rad':phi,'frequencies':frequencies,'ra_now':RAnow,'dec_now':Decnow}
    #     return {'za_rad':theta,'astro_az_rad':phi,'frequencies':frequencies}
    return {'za_rad': theta.transpose(),
            'astro_az_rad': phi.transpose(),
            'frequencies': frequencies,
            'ra_now': RAnow.transpose(),
            'dec_now': Decnow.transpose()}


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
        logger.debug('My Jones correction:\n[%s %s]\n[%s %s]' % (my_j[0, 0], my_j[0, 1], my_j[1, 0], my_j[1, 1]))

        temp = beam_tools.makeUnpolInstrumentalResponse(my_j, my_j)
        logger.debug('My Jones correction:\n[XX:%s XY:%s]\n[YX:%s YY:%s]' % (temp[0, 0], temp[0, 1], temp[1, 0], temp[1, 1]))

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
    logger.warning('Calibration correction factors: XX=%s, XY=%s, YX=%s, YY=%s' % (correction[0, 0],
                                                                                   correction[0, 1],
                                                                                   correction[1, 0],
                                                                                   correction[1, 1]))
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
                # print 'Processing  %s of %s...'%(i,my_shape[2])
            for j in range(my_shape[3]):
                # J1^-1 (Vij (J2^H)^-1)
                b[:, :, i, j] = np.dot(j1inv[:, :, i, j], np.dot(Vij[:, :, i, j], j2Hinv[:, :, i, j]))
    else:
        print 'FIXME for other lengths!!'
        return
    return b


def estimateSkyBrightnessMatrix_SLOW(j1, j2, Vij):
    my_shape = Vij.shape
    b = Vij * 0

    if len(my_shape) == 4:
        for i in range(my_shape[2]):
            for j in range(my_shape[3]):
                j1_2D = j1[:, :, i, j]
                j2_2D = j2[:, :, i, j]
                Vij_2D = Vij[:, :, i, j]

                j1inv = inv2x2(j1_2D)
                j2H = np.transpose(j2_2D.conj())
                j2Hinv = inv2x2(j2H)

                # b_2D = j1_2D * 0
                b_2D = np.dot(j1inv, np.dot(Vij_2D, j2Hinv))
                b[:, :, i, j] = b_2D
    # JUST TEST : b[:,:,i,j]=b_2D*0
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


# convertion from standard MWA coordinate system with X=EW and Y=NS to Thomson,Moran,Swanson (TMS) coordinates where :
# X=S->N, Y=W -> E - see my notes in notebook 
# WARNING : might need to be verified by PSR J0835-4510 
# sign_uv=-1 for AEE and FEE (if -Sigma_P is used in beam_full_EE.py otherwise only Q should have sign flipped)
# sign_q - is only for Analytic model which is different than AEE and FEE (both from FEKO) in terms of sign conventions :
def B2IQUV_TMS(B, sign_uv=1, sign_q=1):
    """Convert sky brightness matrix to I, Q, U, V"""
    B11 = B[0, 0, :, :]
    B12 = B[0, 1, :, :]
    B21 = B[1, 0, :, :]
    B22 = B[1, 1, :, :]

    stokes = {}
    stokes['I'] = (B11 + B22) / 2.
    stokes['Q'] = sign_q * sign_uv * (B11 - B22) / 2.
    stokes['U'] = sign_uv * (B12 + B21) / 2.
    stokes['V'] = sign_uv * 1j * (B21 - B12) / 2.
    return stokes


def parse_options():
    usage = "Usage: %prog [options]\n"
    usage += '\tMeasure Stokes leakage in beam-corrected images\n'
    parser = OptionParser(usage=usage, version=1.00)
    parser.add_option('-o', '--obsid',
                      dest="obsid",
                      default=-1,
                      help="Coma separated list of observations IDs",
                      type="int")
    parser.add_option('-f', '--freq_mhz', '--freq', '--freqs',
                      dest="freq_mhz",
                      default=0,
                      help="Coma separated list of coarse channels",
                      type="float")
    parser.add_option('-d', '--debug',
                      dest="debug",
                      default="1",
                      help="Debug",
                      metavar="STRING",
                      type="int")
    parser.add_option('-m', '--model',
                      dest="model",
                      default="full_EE",
                      help="Model to use",
                      metavar="STRING")
    parser.add_option('--metafits',
                      dest='metafits',
                      default=None,
                      help="FITS file to get delays from (can be metafits)")
    parser.add_option('--xx_file',
                      dest='xx_file',
                      default=None,
                      help="Force xx_file")
    parser.add_option('--yy_file',
                      dest='yy_file',
                      default=None,
                      help="Force yy_file")
    parser.add_option('--xy_file',
                      dest='xy_file',
                      default=None,
                      help="Force xy_file")
    parser.add_option('--xyi_file',
                      dest='xyi_file',
                      default=None,
                      help="Force xyi_file")
    parser.add_option('--out_basename',
                      dest='out_basename',
                      default=None,
                      help="Output file basename")
    parser.add_option('-g', '--gridpoint',
                      dest="gridpoint",
                      default=-1,
                      help="MWA gridpoint where the data was collected [default %default]",
                      type="int")
    parser.add_option('-b', '--beamformer', "--delays",
                      dest='delays',
                      default=None,
                      # default zenith pointing "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"
                      help='16 beamformer delays separated by commas')
    parser.add_option('--h5file',
                      dest='h5file',
                      default="MWA_embedded_element_pattern_V02.h5",
                      help="H5 file")
    parser.add_option('--wsclean',
                      action="store_true",
                      dest="wsclean_image",
                      default=True,
                      help="Reorders final components to bring the signs back to IEEE or Thomson,Moran,Swanson (TMS) convention from WSCLEAN-MWA X=EW,Y=NS convention (non-standard)")
    parser.add_option('-r', '--rts',
                      action="store_true",
                      dest="rts_image",
                      default=False,
                      help="If it is RTS image -> XX needs to be swapped with YY in Vij")
    parser.add_option('-z','--zenith_norm',
                      action="store_true",
                      dest="zenithnorm",
                      default=False,
                      help="If normalise to zenith [default %]")                      

    (options, args) = parser.parse_args()

    return (options, args)


if __name__ == "__main__":
    my_dir = ''  # Should work ok when run from directory where files are.

    (options, args) = parse_options()

    logger.setLevel(logging.DEBUG)
    # Pick image & gridpoint
    # hogbom appears to be less noisey than clarkstokes.

    delays = None
    # if nothing set for delays in options, but metafits is provided -> use info from metafits
    if options.metafits is not None and options.delays is None:
        print "here ???"
        try:
            f = pyfits.open(options.metafits)
        except Exception, e:
            logger.error('Unable to open FITS file %s: %s' % (options.metafits, e))
            sys.exit(1)
        if 'DELAYS' not in f[0].header.keys():
            logger.error('Cannot find DELAYS in %s' % options.metafits)
            sys.exit(1)
        options.delays = f[0].header['DELAYS']
        print "DEBUG : options.delays = %s" % (options.delays)
        try:
            delays = [int(x) for x in options.delays.split(',')]
        except Exception, e:
            logger.error('Unable to parse beamformer delays %s: %s' % (options.delays, e))
            sys.exit(1)

    # if still nothing in options.delays (metafits not provided) -> use zenith delays
    if options.delays is None:
        options.delays = "0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0"

    # parse whatever is in options.delays :
    if options.delays is not None and delays is None:
        print "Parsing options.delays = |%s|" % (options.delays)
        try:
            if (',' in options.delays):
                delays = map(int, options.delays.split(','))
            else:
                delays = 16 * [int(options.delays)]
        except Exception:
            logger.error("Could not parse beamformer delays %s\n" % options.delays)
            sys.exit(1)

    gridpoint = -1
    if options.gridpoint > 0:
        gridpoint = options.gridpoint

    # find delays by gridpoint if given as parameter:
    if gridpoint >= 0:
        print "Getting delays for gridpoint = %d" % (gridpoint)
        delays = mwa_sweet_spots.get_delays(gridpoint)

    if options.model not in ['analytic', 'advanced', 'full_EE', 'full_EE_AAVS05', 'FEE', 'Full_EE', '2016', '2015', '2014']:
        logger.error("Model %s not found\n" % options.model)
        sys.exit(1)

    logger.info("########################################")
    logger.info("PARAMETERS:")
    logger.info("########################################")
    logger.info("debug level = %d" % (options.debug))
    logger.info("Model    = %s" % (options.model))
    logger.info("Metafits      = %s" % (options.metafits))
    logger.info("Externally provided file names are xx/xy/xyi/yy = %s/%s/%s/%s" % (
        options.xx_file, options.xy_file, options.xyi_file, options.yy_file))
    logger.info("H5 file   = %s" % options.h5file)
    logger.info("Gridpoint = %d" % gridpoint)
    logger.info("Delays    = %s" % delays)
    logger.info("wsclean_image = %s" % options.wsclean_image)
    logger.info("RTS_image     = %s" % options.rts_image)
    logger.info("Obsid         = %d" % options.obsid)
    logger.info("Freq          = %.2f [MHz]" % options.freq_mhz)
    logger.info("Zenith norm   = %s" % options.zenithnorm)
    logger.info("########################################")

    model = options.model
    xx_file = options.xx_file
    xy_file = options.xy_file
    xyi_file = options.xyi_file
    yy_file = options.yy_file

    xx = pyfits.open(xx_file)
    yy = pyfits.open(yy_file)
    xy = None
    if xy_file is not None:
        xy = pyfits.open(xy_file)

    xyi = None
    if xyi_file is not None:
        xyi = pyfits.open(xyi_file)

    h = xx[0].header
    naxes = h['NAXIS']

    data_xx = None
    data_xy = None
    data_xyi = None
    data_yy = None

    # updated based on full_Stokes_beam_correct_xxyy.py (use xy=xyi=0 if they are not provided) :
    if naxes >= 4:
        data_xx = xx[0].data[0, 0]
    else:
        data_xx = xx[0].data
    xx.close()

    if naxes >= 4:
        data_yy = yy[0].data[0, 0]
    else:
        data_yy = yy[0].data
    yy.close()

    if xy is not None:
        if naxes >= 4:
            data_xy = xy[0].data[0, 0]
        else:
            data_xy = xy[0].data
        xy.close()
    else:
        data_xy = data_xx * 0.00
        print "WARNING : --xy_file not provided -> using ZERO XY image"

    if xyi is not None:
        if naxes >= 4:
            data_xyi = xyi[0].data[0, 0]
        else:
            data_xyi = xyi[0].data
        xyi.close()
    else:
        data_xyi = data_xx * 0.00
        print "WARNING : --xyi_file not provided -> using ZERO XYi image"

    Joneses = {}
    stokeses = {}

    logger.info('Processing %s' % xx_file)
    beam_info = get_azza_from_fits(xx_file, gps=options.obsid, freq_mhz=options.freq_mhz)
    if beam_info is None:
        logger.error("Could not generate azza map for file %s - probably missing information" % xx_file)
        sys.exit(-1)

    target_freq_Hz = beam_info['frequencies'][0]
    az = beam_info['astro_az_rad']
    za = beam_info['za_rad']
    az_original = np.copy(az)
    za_original = np.copy(za)
    ra_now = beam_info['ra_now']
    dec_now = beam_info['dec_now']
    # print "DEBUG : beam_info.shape = %d" % beam_info.shape()

    # visibilites constructed from instrumental polarisation images from WSCLEAN:
    Vij = np.array([[data_xx, data_xy + 1j * data_xyi], [data_xy - 1j * data_xyi, data_yy]])
    if options.rts_image:
        Vij = np.array([[data_yy, data_xy + 1j * data_xyi], [data_xy - 1j * data_xyi, data_xx]])
    # delays = beam_tools.gridpoint2delays(gridpoint, os.path.join(MWAtools_pb_dir, 'MWA_sweet_spot_gridpoints.csv'))

    print "delays = %s , frequency = %.2f Hz" % (delays, target_freq_Hz)

    sign_uv = 1
    sign_q = 1
    if model == 'full_EE' or model == '2016' or model == 'FEE' or model == 'Full_EE':
        logger.info("Correcting with full_EE(%s) model at frequency %.2f Hz" % (model, target_freq_Hz))
        # print "DEBUG = %s" % (az)
        Jones = primary_beam.MWA_Tile_full_EE(za, az,
                                              target_freq_Hz,
                                              delays=delays,
                                              zenithnorm=options.zenithnorm,
                                              jones=True,
                                              interp=True)

        if options.wsclean_image:
            sign_uv = -1
    elif model == 'avg_EE' or model == 'advanced' or model == '2015' or model == 'AEE':
        logger.info("Correcting with AEE(%s) model at frequency %.2f Hz" % (model, target_freq_Hz))
        # logging.getLogger("mwa_tile").setLevel(logging.DEBUG)
        Jones = primary_beam.MWA_Tile_advanced(za, az,
                                               target_freq_Hz,
                                               delays=delays,
                                               zenithnorm=options.zenithnorm,
                                               jones=True)
        if options.wsclean_image:
            sign_uv = -1
    elif model == 'analytic' or model == '2014':
        logger.info("Correcting with analytic model")
        Jones = primary_beam.MWA_Tile_analytic(za, az,
                                               target_freq_Hz,
                                               delays=delays,
                                               zenithnorm=options.zenithnorm,
                                               jones=True)
        if options.wsclean_image:
            sign_q = -1
    else:
        logger.error('Unknown model: %s' % model)
        sys.exit()

    # Swap axes so that Jones 2x2 matrix forms the first 2 dimensions
    Jones = np.swapaxes(np.swapaxes(Jones, 0, 2), 1, 3)
    Joneses[model] = Jones  # Keep for further analysis

    # visibilities corrected for Direction Independed Gain (DIG) - it is in former step in CASA:
    Vij_corrected = Vij  # Skip beam correction of cal solutions - this is done in CASA

    logger.info('Estimate brightness matrix... %s' % (datetime.datetime.now().time()))
    B = estimateSkyBrightnessMatrix_SLOW(Jones, Jones, Vij_corrected)

    logger.info('Get Stokes... %s , sign_uv=%d, sign_q=%d' % (datetime.datetime.now().time(), sign_uv, sign_q))
    stokes = B2IQUV_TMS(B, sign_uv=sign_uv, sign_q=sign_q)

    stokeses[model] = stokes  # Keep for further analysis

    # Also write individual paramces
    stokes_params = ['I', 'Q', 'U', 'V']
    for i in range(4):
        s = stokes_params[i]
        # Save to new file
        f_single = pyfits.open(xx_file)  # Use IQUV image as template
        comment = 'Extracted Stokes %s image a 2D array' % (stokes_params[i])

        # Save single stokes to new file
        out_filename_single = s + '.fits'
        if options.out_basename is not None:
            out_filename_single = options.out_basename + "_" + s + '.fits'

        # is ABS because I,Q,U,V are complex and we need real FITS files ?
        # YES : but imaginary part is really very small
        # changed abs -> real
        f_single[0].data = np.real(stokes[s])

        print 'Writing %s' % out_filename_single
        f_single.writeto(out_filename_single, clobber=True)
        f_single.close()
