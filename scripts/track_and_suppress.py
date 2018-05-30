#!/usr/bin/env python

"""
Script finds optimal pointing so that beam in one source direction is maximised and in the direction of source 2 is minimised
developed specifically for Elaine S. observations of TN0924-2201 which is ~10 deg from HydA

Starting version by Marcin Sokolowski

This is the script interface to the functions and modules defined in MandC_Core/pb/primarybeamap.py

"""

import getpass
import logging
from optparse import OptionParser
import sys

import astropy
from astropy.coordinates import SkyCoord
from astropy.time import Time

from mwa_pb import config
from mwa_pb.suppress import get_best_gridpoints_supress_sun, get_best_gridpoints

# configure the logging
logging.basicConfig(filename='/tmp/suppress.log', format='# %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger('pb.track_and_suppress')
logger.setLevel(logging.DEBUG)

if __name__ == '__main__':
    usage = "Usage: %prog [options]\n"
    usage += "\tGenerates single_observation.py commands to track a given source while keeping a given source in a null.\n"

    parser = OptionParser(usage=usage)
    parser.add_option('-g', '--gps',
                      dest='gps_start',
                      default=0,
                      help='GPS time',
                      type="int")
    parser.add_option('-u', '--unixtime', '--ux',
                      dest='ux_start',
                      default=0,
                      help='Start unix time',
                      type="int")
    parser.add_option('--step',
                      dest='step',
                      default=120,
                      help='Individual observation length in seconds, default %default',
                      type="int")
    parser.add_option('--duration',
                      dest='duration',
                      default=36000,
                      help='Observing duration in seconds, default %default',
                      type="int")
    parser.add_option('-m', '--model',
                      dest='model',
                      default='analytic',
                      help='beam model: analytic, advanced, full_EE, full_EE_AAVS05 (default %default)')
    parser.add_option('--obs_source_ra_deg',
                      dest='obs_source_ra_deg',
                      default=0.0,
                      help='Observations source RA[deg]',
                      type=float)
    parser.add_option('--obs_source_dec_deg',
                      dest='obs_source_dec_deg',
                      default=0.0,
                      help='Observations source DEC[deg]',
                      type=float)

    parser.add_option('--avoid_source_ra_deg',
                      dest='avoid_source_ra_deg',
                      default=None,
                      help='Avoided source RA[deg] - default Sun',
                      type=float)
    parser.add_option('--avoid_source_dec_deg',
                      dest='avoid_source_dec_deg',
                      default=None,
                      help='Avoided source DEC[deg] - default Sun',
                      type=float)

    parser.add_option('--max_beam_distance_deg',
                      dest='max_beam_distance_deg',
                      default=30,
                      help='Maximum distance from the two sources in degrees, default %default',
                      type=float)
    parser.add_option('--min_gain',
                      dest='min_gain',
                      default=0.2,
                      help='Minimum acceptable gain in the direction of object',
                      type=float)

    # observation parameters :
    parser.add_option('-o', '--outfile', '--out',
                      dest="outfile",
                      default=None,
                      help="Name of output file [default %default]")
    parser.add_option('-c', '--channel',
                      dest='channel',
                      default=175,
                      help='Center channel(s) of observation, default %default',
                      type="int")
    parser.add_option('--inttime',
                      dest='inttime',
                      default=2,
                      help='Integration time [default %default sec]',
                      type="float")
    parser.add_option('--freqres',
                      dest='freqres',
                      default=10,
                      help='Frequency resolution [default %default kHz]',
                      type="int")
    parser.add_option('--radec',
                      dest='radec',
                      action='store_true',
                      help='Use RA and DEC for coordinates in single_observation.py, instead of alt/az values',
                      default=False)
    parser.add_option('--object',
                      dest='obj',
                      default="TN0924",
                      help='Object name [default %default]')
    parser.add_option('--obsname',
                      dest='obsname',
                      default="Test",
                      help='Observation name [default %default]')
    parser.add_option('--project',
                      dest='project',
                      default="C001",
                      help='Project name [default %default - testing]')

    parser.add_option('-v', '--verbose',
                      action="store_true",
                      dest="verbose",
                      default=False,
                      help="Increase verbosity of output")

    (options, args) = parser.parse_args()

    print "######################################################"
    print "PARAMETERS:"
    print "######################################################"
    print "min_gain = %.4f" % (options.min_gain)
    print "######################################################"

    model = options.model
    if model not in ['analytic', 'advanced', 'full_EE', 'full_EE_AAVS05']:
        logger.error("Model %s not found\n" % model)
        sys.exit(1)

    if (options.ux_start and options.gps_start):
        logger.critical("Must provide either GPS or Unix format start time, not both")
        sys.exit(-1)

    if options.ux_start:
        t = Time(options.ux_start, format='unix', scale='utc')
        tgps = 8 * int((t.gps + 7) / 8)  # Always round up to the next modulo 8 second, not down
        start_time = Time(tgps, format='gps', scale='utc')
    elif options.gps_start:
        tgps = 8 * int((options.gps_start + 7) / 8)  # Always round up to the next modulo 8 second, not down
        start_time = Time(tgps, format='gps', scale='utc')
    else:
        logger.critical("Must provide either GPS or Unix format start time")
        sys.exit(-1)

    step = 8 * int((options.step + 7) / 8)  # Always round up to the next modulo 8 second, not down

    if (options.avoid_source_dec_deg is None) or (options.avoid_source_dec_deg is None):
        tracklist = get_best_gridpoints_supress_sun(gps_start=start_time.gps + 16,  # Add 16 to allow for mode change
                                                    obs_source_ra_deg=options.obs_source_ra_deg,
                                                    obs_source_dec_deg=options.obs_source_dec_deg,
                                                    model=options.model,
                                                    min_gain=options.min_gain,
                                                    max_beam_distance_deg=options.max_beam_distance_deg,
                                                    channel=options.channel,
                                                    verb_level=1,
                                                    duration=options.duration,
                                                    step=step)
    else:
        tracklist = get_best_gridpoints(gps_start=start_time.gps + 16,  # Add 16 to allow for mode change
                                        obs_source_ra_deg=options.obs_source_ra_deg,
                                        obs_source_dec_deg=options.obs_source_dec_deg,
                                        avoid_source_ra_deg=options.avoid_source_ra_deg,
                                        avoid_source_dec_deg=options.avoid_source_dec_deg,
                                        model=options.model,
                                        min_gain=options.min_gain,
                                        max_beam_distance_deg=options.max_beam_distance_deg,
                                        channel=options.channel,
                                        verb_level=1,
                                        duration=options.duration,
                                        step=step)

    start_time.location = config.MWAPOS

    if options.outfile is None:
        # 20170216_cmdfile_J112537_LST9.91.txt
        outfile_name = ("%s_cmdfile_%s_LST%.4f_BeamModel-%s_mingain%.2f.txt" % (start_time.datetime.strftime("%Y%m%d"),
                                                                                options.obj,
                                                                                start_time.sidereal_time('mean').hour,
                                                                                options.model,
                                                                                options.min_gain))
    else:
        outfile_name = options.outfile

    creator = getpass.getuser()

    outfile = open(outfile_name, 'a+')
    outfile.write("# setting correlator mode :\n")

    if options.inttime >= 1.0:
        inttime_str = '%d' % int(options.inttime)
    else:
        inttime_str = '%.1f' % (options.inttime)

    command = "single_observation.py --creator=%(creator)s --starttime=%(otime)d --stoptime=++16s --freq=69,24 "
    command += "--obsname=CORR_MODE_%(freqres)d_%(inttime)s --inttime=%(inttime)s --freqres=%(freqres)d "
    command += "--az=0.0 --el=90.0 --mode=CORR_MODE_CHANGE --project=%(project)s\n"
    outfile.write(command % {'creator': creator,
                             'otime': start_time.gps,
                             'channel': options.channel,
                             'inttime': inttime_str,
                             'freqres': options.freqres,
                             'project': options.project})
    outfile.write("\n")

    for entry in tracklist:
        otime, step, az, alt = entry
        obstime = Time(otime, format='gps', scale='utc')
        pos = SkyCoord(az, alt, frame='altaz', unit=(astropy.units.deg, astropy.units.deg), location=config.MWAPOS,
                       obstime=obstime)
        posrd = pos.transform_to('icrs')
        if options.radec:
            command = "single_observation.py --creator=%(creator)s --starttime=%(otime)s --stoptime=++%(step)s --freq=%(channel)d,24 "
            command += "--obsname=%(obj)s_%(channel)d --inttime=%(inttime)s --freqres=%(freqres)d --useazel --usegrid= "
            command += "--ra=%(ra).4f --dec=%(dec).4f --project=%(project)s\n"
        else:
            command = "single_observation.py --creator=%(creator)s --starttime=%(otime)s --stoptime=++%(step)s --freq=%(channel)d,24 "
            command += "--obsname=%(obj)s_%(channel)d --inttime=%(inttime)s --freqres=%(freqres)d --useazel --usegrid= "
            command += "--azimuth=%(az).4f --elevation=%(alt).4f --project=%(project)s\n"

        outfile.write(command % {'creator': creator,
                                 'otime': otime,
                                 'step': step,
                                 'channel': options.channel,
                                 'obj': options.obj,
                                 'inttime': inttime_str,
                                 'freqres': options.freqres,
                                 'ra': posrd.ra.deg,
                                 'dec': posrd.dec.deg,
                                 'alt': alt,
                                 'az': az,
                                 'project': options.project})

    print "Script saved to file %s" % (outfile_name)
