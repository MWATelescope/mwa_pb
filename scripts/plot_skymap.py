#!/usr/bin/env python

"""
Produce a sky map showing the beam shape for a given observation.
"""

from optparse import OptionParser
import sys
import time

from astropy.time import Time

from mwa_pb import skymap
from mwa_pb import metadata


##################################################
if __name__ == "__main__":
    usage = "Usage: %prog [options]\n"
    usage += '\tMakes plot of MWA sky, showing the primary beam.\n'
    parser = OptionParser(usage=usage)

    parser.add_option('-o', '--out', dest='out', default='mwasky.png',
                      help='Name for output [default=%default]')
    parser.add_option('-s', '--obsid', dest='obsid', default=None,
                      help='Observation ID to process [default=most recent]')
    parser.add_option('-v', '--viewtime', dest='viewtime', default=None,
                      type='int',
                      help='Time in GPS seconds for sky map display [default=now]')
    parser.add_option('-c', '--channel', dest='channel', default=None,
                      type='int',
                      help='Coarse channel to use to calculate beam shape (defaults to observation centre channel')
    parser.add_option('-p', '--plotscale', dest='plotscale', default=1.0,
                      type='float',
                      help='Scale factor for plot size (default=1.0 for %d x %d)' % (skymap.FIGSIZE * skymap.DPI,
                                                                                     skymap.FIGSIZE * skymap.DPI))
    parser.add_option('--hidenulls', dest='hidenulls', default=False,
                      action='store_true',
                      help='Do not include black contours for nulls on plot.')
    parser.add_option('--hidegleamsources', dest='gleamsources', default=True,
                      action='store_false',
                      help='Do not show dots for GLEAM sources on the plot.')
    parser.add_option('--hideconstellations', dest='constellations', default=True,
                      action='store_false',
                      help='Do not show constellation stick figures on the plot.')
    parser.add_option('--notext', dest='notext', default=False,
                      action='store_true',
                      help='Do not include text with target details on the plot?')

    parser.add_option('--daemon', '-d', dest='daemon', default=False,
                      action='store_true',
                      help='Run as a daemon, generating a new sky map every few minutes')
    parser.add_option('-b', '--background', dest='background', default='black',
                      help='Background colour, eg red, green, transparent [default=%default]')
    (options, args) = parser.parse_args()

    skydata = skymap.SkyData()
    if skydata is None:
        sys.exit(1)

    if options.daemon:
        while True:
            obsinfo = metadata.get_observation()
            if obsinfo.starttime <= Time.now().gps < (obsinfo.starttime + 300):
                observing = True
            else:
                observing = False
            result = skymap.plot_MWAconstellations(obsinfo=obsinfo,
                                                   outfile=options.out,
                                                   notext=options.notext,
                                                   observing=observing,
                                                   skydata=skydata,
                                                   showbeam=True,
                                                   constellations=options.constellations,
                                                   gleamsources=options.gleamsources,
                                                   background=options.background,
                                                   hidenulls=options.hidenulls,
                                                   channel=options.channel,
                                                   xmas=False,
                                                   plotscale=options.plotscale,
                                                   )
            skymap.plt.close('all')
            time.sleep(60)
    else:
        obsinfo = metadata.get_observation(options.obsid)
        if not obsinfo:
            print('No observation info to display.')
            sys.exit(1)

        result = skymap.plot_MWAconstellations(obsinfo=obsinfo,
                                               outfile=options.out,
                                               notext=options.notext,
                                               viewgps=options.viewtime,
                                               observing=True,
                                               skydata=skydata,
                                               showbeam=True,
                                               constellations=options.constellations,
                                               gleamsources=options.gleamsources,
                                               background=options.background,
                                               hidenulls=options.hidenulls,
                                               channel=options.channel,
                                               xmas=False,
                                               plotscale=options.plotscale,
                                               )
    if result is None:
        sys.exit(1)
    else:
        sys.exit(0)


######################################################################
