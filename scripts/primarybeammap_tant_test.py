#!/usr/bin/env python
"""
  Script calling function primarybeammap_tant.py to calculate antenna temperature according to MWA beam model (analytic, AEE or FEE) and scaled Haslam map
  Example usage:
  python ./primarybeammap_tant_test.py -c 169 -p all -g 0  -m analytic
  python ./primarybeammap_test.py -b 18,13,8,3,17,12,7,2,16,11,6,1,15,10,5,0 -c 169 -p all -g 0  -m full_EE

  main task is:
  make_primarybeammap()

  This is the script interface to the functions and modules defined in MWA_Tools/src/primarybeamap.py
"""

import errno
import os
import sys
from optparse import OptionParser

import numpy

from mwa_pb.primarybeammap_tant import logger, contourlevels, make_primarybeammap


def mkdir_p(path):
  try:
    os.makedirs(path)
  except OSError as exc:  # Python >2.5
    if exc.errno == errno.EEXIST:
      pass
    else:
      raise


def main():
  usage = "Usage: %prog [options]\n"
  usage += "\tCreates an image of the 408 MHz sky (annoted with sources) that includes contours for the MWA primary beam\n"
  usage += "\tThe beam is monochromatic, and is the sum of the XX and YY beams\n"
  usage += "\tThe date/time (UT) and beamformer delays must be specified\n"
  usage += "\tBeamformer delays should be separated by commas\n"
  usage += "\tFrequency is in MHz, or a coarse channel number (can also be comma-separated list)\n"
  usage += "\tDefault is to plot centered on RA=0, but if -r/--racenter, will center on LST\n"
  usage += "\tContours will be plotted at %s of the peak\n" % contourlevels
  usage += "\tExample:\tpython primarybeammap.py -c 98 --beamformer=1,0,0,0,3,3,3,3,6,6,6,6,9,9,9,8 \n\n"

  parser = OptionParser(usage=usage)
  parser.add_option('-c', '--channel', dest='channel', default=None,
                    help='Center channel(s) of observation')
  parser.add_option('-f', '--frequency', dest='frequency', default=None,
                    help='Center frequency(s) of observation [MHz]')
  parser.add_option('-b', '--beamformer', dest='delays', default="0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0",
                    help='16 beamformer delays separated by commas')
  parser.add_option('-D', '--date', dest='date', default=None,
                    help='UT Date')
  parser.add_option('-t', '--time', dest='time', default=None,
                    help='UT Time')
  parser.add_option('-g', '--gps', dest='gps', default=None,
                    help='GPS time')
  parser.add_option('-m', '--model', dest='model', default='analytic',
                    help='beam model: analytic, advanced, full_EE, full_EE_AAVS05')
  parser.add_option('-p', '--plottype', dest='plottype', default='beamsky',
                    help='Type of plot: all, beam, sky, beamsky, beamsky_scaled')
  parser.add_option('--title', dest='title', default=None,
                    help='Plot title')
  parser.add_option('-e', '--ext', dest='extension', default='png',
                    help='Plot extension [default=%default]')
  parser.add_option('-r', '--racenter', action="store_true", dest="center", default=False,
                    help="Center on LST?")
  parser.add_option('-s', '--sunline', dest="sunline", default="1", choices=['0', '1'],
                    help="Plot sun [default=%default]")
  parser.add_option('--tle', dest='tle', default=None,
                    help='Satellite TLE file')
  parser.add_option('--duration', dest='duration', default=300, type=int,
                    help='Duration for plotting satellite track')
  parser.add_option('--size', dest='size', default=1000, type=int,
                    help='Resolution of created beam file')
  parser.add_option('--dir', dest='dir', default=None, help='output directory')

  parser.add_option('-v', '--verbose', action="store_true", dest="verbose", default=False,
                    help="Increase verbosity of output")

  (options, args) = parser.parse_args()

  if options.dir is not None:
    mkdir_p(options.dir)

  if options.frequency is not None:
    if (',' in options.frequency):
      try:
        frequency = map(float, options.frequency.split(','))
      except ValueError:
        logger.error("Could not parse frequency %s\n" % options.frequency)
        sys.exit(1)
    else:
      try:
        frequency = float(options.frequency)
      except ValueError:
        logger.error("Could not parse frequency %s\n" % options.frequency)
        sys.exit(1)
  else:
    frequency = options.frequency
  if options.channel is not None:
    if (',' in options.channel):
      try:
        channel = map(float, options.channel.split(','))
      except ValueError:
        logger.error("Could not parse channel %s\n" % options.channel)
        sys.exit(1)
    else:
      try:
        channel = float(options.channel)
      except ValueError:
        logger.error("Could not parse channel %s\n" % options.channel)
        sys.exit(1)
  else:
    channel = options.channel

  if options.delays is not None:
    try:
      if (',' in options.delays):
        delays = map(int, options.delays.split(','))
      else:
        delays = 16 * [int(options.delays)]
    except ValueError:
      logger.error("Could not parse beamformer delays %s\n" % options.delays)
      sys.exit(1)
  else:
    delays = options.delays

  extension = options.extension
  plottype = options.plottype
  model = options.model
  if model not in ['analytic', 'advanced', 'full_EE', 'full_EE_AAVS05']:
    logger.error("Model %s not found\n" % model)
    sys.exit(1)
  if plottype not in ['all', 'beam', 'sky', 'beamsky', 'beamsky_scaled']:
    logger.error("Plot type %s not found\n" % plottype)
    sys.exit(1)
  gpsstring = options.gps
  gps = int(gpsstring)

  if (len(delays) < 16):
    logger.error("Must supply 1 or 16 delays\n")
    sys.exit(1)
  if (frequency is None):
    if (channel is not None):
      if (isinstance(channel, list)):
        frequency = list(1.28 * numpy.array(channel))  # multiplication by 1e6 is done later at line Convert to Hz
      else:
        frequency = 1.28 * channel  # multiplication by 1e6 is done later at line Convert to Hz
  if frequency is None:
    logger.error("Must supply frequency or channel\n")
    sys.exit(1)
  if (isinstance(frequency, int) or isinstance(frequency, float)):
    frequency = [frequency]
  frequency = numpy.array(frequency) * 1e6  # Convert to Hz

  for freq in frequency:
    print 'frequency', freq
    result = make_primarybeammap(gps, delays, freq, model=model,
                                 plottype=plottype, extension=extension, resolution=options.size, directory=options.dir)
    if (result is not None):
      print "Wrote %s" % result


if __name__ == "__main__":
  main()
