#! /usr/bin/env python
"""
    This is an extension of MWA_Tools/scripts/make_beam.py to enable new spherical harmonics beam model:
    e.g.
    python make_beam_test.py -f image_name.fits -d 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 -v --full_EE

    plock[pbtest]% python ~/mwa/bin/make_beam.py -f P00_w.fits -v
    # INFO:make_beam: Computing for 2011-09-27 14:05:06+00:00
    # INFO:make_beam: Created primary beam for 154.24 MHz and delays=0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
    # INFO:make_beam: XX beam written to P00_w_beamXX.fits
    # INFO:make_beam: YY beam written to P00_w_beamYY.fits

"""
# degub :
import pdb

import logging
from optparse import OptionParser
import sys

import astropy.io.fits as pyfits

from mwa_pb import config
from mwa_pb import make_beam

# configure the logging
logging.basicConfig(format='# %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('make_beam')
logger.setLevel(logging.DEBUG)


######################################################################
def main():
  usage = "Usage: %prog [options]\n"
  usage += '\tMakes primary beams associated with a FITS image\n'
  parser = OptionParser(usage=usage, version=config.__version__)
  parser.add_option('-f', '--filename',
                    dest="filename",
                    default=None,
                    help="Create primary beam for <FILE>",
                    metavar="FILE")
  parser.add_option('-e', '--ext',
                    dest='ext',
                    type=str,
                    default='0',
                    help='FITS extension name or number [default=%default]')
  parser.add_option('-m', '--metafits',
                    dest='metafits',
                    default=None,
                    help="FITS file to get delays from (can be metafits)")
  parser.add_option('-d', '--delays',
                    dest="delays",
                    default=None,
                    help="Beamformer delays to use; 16 comma-separated values")
  #    parser.add_option('--analytic',action="store_true",dest="analytic_model",default=False,
  #                      help="Use the old analytic dipole model, instead of the default Sutinjo 2014 model.")
  #    parser.add_option('--full_EE',action="store_true",dest="full_EE_model",default=False,
  #                      help="Use the new full embedded element model (V02), instead of the default Sutinjo 2014 model.")
  parser.add_option('--model',
                    dest="model",
                    default="2014",
                    help="Model to be used : analytic, 2014, 2016 [default=%default]")
  parser.add_option('--jones',
                    dest='jones',
                    default=False,
                    action='store_true',
                    help="Compute Jones matrix instead of power beam? [default=False]")
  #    parser.add_option('--height',dest='height',default=primary_beam._DIPOLE_HEIGHT,
  #                      type=float,
  #                      help='Dipole height (m) (only an option for analytic beam model) [default=%default]')
  #    parser.add_option('--sep',dest='separation',default=primary_beam._DIPOLE_SEPARATION,
  #                      type=float,
  #                      help='Dipole separation (m) (only an option for analytic beam model) [default=%default]')
  parser.add_option('-v', '--verbose',
                    action="store_true",
                    dest="verbose",
                    default=False,
                    help="Increase verbosity of output")
  parser.add_option('-g','--gridpoint',dest="gridpoint",default=-1, help="MWA gridpoint where the data was collected [default %default]",type="int")
  parser.add_option('-o','--obsid',dest="obsid",default=-1, help="Coma separated list of observations IDs",type="int")
  parser.add_option('--freq_cc',dest="freq_cc",default=0, help="Coma separated list of coarse channels",type="int")
  parser.add_option('--freq_mhz',dest="freq_mhz",default=0, help="Coma separated list of coarse channels",type="float")

  # just to keep backword compatibility :
  parser.add_option('--analytic',action="store_true",dest="analytic_model",default=False, help="Use the old analytic dipole model, instead of the default Sutinjo 2014 model.")
  parser.add_option('--full_EE',action="store_true",dest="full_EE_model",default=False, help="Use the new full embedded element model (V02), instead of the default Sutinjo 2014 model.")



  (options, args) = parser.parse_args()

  if (options.verbose):
    logger.setLevel(logging.INFO)

    if options.full_EE_model :
       options.model = '2016'
    if options.analytic_model :
       options.model = '2014'

  print "###########################################"
  print "PARAMETERS:"
  print "###########################################"
  print "Filename = %s" % options.filename
  print "obsid    = %d" % options.obsid
  print "model    = %s" % options.model
  print "metafits = %s" % options.metafits
  print "delays   = %s" % options.delays
  print "###########################################"

  if options.model not in ['analytic','advanced','full_EE', 'full_EE_AAVS05','FEE','Full_EE','2016','2015','2014']:
    logger.error("Model %s not found\n" % model)
    sys.exit(1)

  try:
    extnum = int(options.ext)
    ext = extnum
  except ValueError:
    ext = options.ext
    pass

  delays = None
  if options.delays is not None:
    try:
      delays = [int(x) for x in options.delays.split(',')]
    except Exception, e:
      logger.error('Unable to parse beamformer delays %s: %s' % (options.delays, e))
      sys.exit(1)

  if options.metafits is not None:
    try:
      f = pyfits.open(options.metafits)
    except Exception, e:
      logger.error('Unable to open FITS file %s: %s' % (options.metafits, e))
      sys.exit(1)
    if 'DELAYS' not in f[0].header.keys():
      logger.error('Cannot find DELAYS in %s' % options.metafits)
      sys.exit(1)
    delays = f[0].header['DELAYS']
    try:
      delays = [int(x) for x in delays.split(',')]
    except Exception, e:
      logger.error('Unable to parse beamformer delays %s: %s' % (delays, e))
      sys.exit(1)

  if options.filename is None:
    logger.error('Must supply a filename')
    sys.exit(1)

  if options.obsid is None or options.obsid <= 0:
    logger.warning("ObsID not provided !")
    if options.metafits is not None:
        logger.warning("Will try to use first 10 digits of the metafits file name %s" % options.metafits)
        obsid_str = options.metafits[0:10]
        if int(obsid_str) > 0:
            options.obsid = int(obsid_str)
            logger.warning("Obsid from metafits file %s is %d" % (options.metafits, options.obsid))
        else:
            logger.error("Could not obtain obsID from metafits file name %s -> cannot continue" % options.metafits)
            sys.exit(1)

  if options.delays is not None :
     delays = options.delays
  # find delays by gridpoint if given as parameter:
  if options.gridpoint >= 0:
    print
    "Getting delays for gridpoint = %d" % (options.gridpoint)
    delays_xy = mwa_sweet_spots.get_delays(options.gridpoint)
    delays = delays_xy[0]
    print
    "Delays for gridpoint = %d are : %s" % (options.gridpoint, delays)

  if delays is None:
    logger.error("Must provide delays, or a metafits file.")
    sys.exit(1)

  #    out=make_beam.make_beam(options.filename, ext=ext, delays=options.delays,
  #                            analytic_model=options.analytic_model,
  #                            jones=options.jones,
  #                            precess=options.precess,
  #                            dipheight=options.height, dip_sep=options.separation,
  #                            full_EE_model=options.full_EE_model)

  out = make_beam.make_beam(options.filename, ext=ext, delays=delays,
                            jones=options.jones,
                            model=options.model,
                            gps = options.obsid,
                            freq_mhz = options.freq_mhz)

  if out is None:
    logger.error('Problem creating primary beams')
    sys.exit(1)

  sys.exit(0)


######################################################################

if __name__ == "__main__":
  main()
