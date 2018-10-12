"""
Script finds optimal pointing so that beam in one source direction is maximised and in the direction of source 2 is minimised 
developed specifically for Elaine S. observations of TN0924-2201 which is ~10 deg from HydA 

Starting version by Marcin Sokolowski

main task is:
make_primarybeammap()

This is the script interface to the functions and modules defined in MWA_Tools/src/primarybeamap.py

"""

import logging

import numpy

import astropy
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time

import config
import primarybeammap_tant
import mwa_sweet_spots

# configure the logging
logging.basicConfig()
logger = logging.getLogger('pb.suppress')
logger.setLevel(logging.DEBUG)


def get_best_gridpoints(gps_start,
                        obs_source_ra_deg,
                        obs_source_dec_deg,
                        avoid_source_ra_deg,
                        avoid_source_dec_deg,
                        model="analytic",
                        min_gain=0.1,
                        max_beam_distance_deg=360,
                        channel=145,
                        verb_level=1,
                        duration=3600,
                        step=120):
    frequency = channel * 1.28

    stime = Time(gps_start, format='gps', scale='utc')

    if model not in ['analytic', 'advanced', 'full_EE', 'full_EE_AAVS05']:
        logger.error("Model %s not found\n" % model)

    gp_numbers = mwa_sweet_spots.all_grid_points.keys()
    gp_numbers.sort()
    gp_azes = numpy.array([mwa_sweet_spots.all_grid_points[i][1] for i in gp_numbers])
    gp_alts = numpy.array([mwa_sweet_spots.all_grid_points[i][2] for i in gp_numbers])
    gp_delays = [mwa_sweet_spots.all_grid_points[i][4] for i in gp_numbers]

#    gp_positions = astropy.coordinates.AltAz(az=astropy.coordinates.Angle(gp_azes, unit=astropy.units.deg),
#                                             alt=astropy.coordinates.Angle(gp_alts, unit=astropy.units.deg),
#                                             location=config.MWAPOS)
    obs_source = SkyCoord(ra=obs_source_ra_deg,
                          dec=obs_source_dec_deg,
                          equinox='J2000',
                          unit=(astropy.units.deg, astropy.units.deg))
    obs_source.location = config.MWAPOS

    avoid_source = SkyCoord(ra=avoid_source_ra_deg,
                            dec=avoid_source_dec_deg,
                            equinox='J2000',
                            unit=(astropy.units.deg, astropy.units.deg))
    avoid_source.location = config.MWAPOS

    freq = frequency * 1e6
    tracklist = []  # List of (starttime, duration, az, el) tuples
    for starttime in range(int(stime.gps), int(stime.gps + duration), int(step)):
        t = Time(starttime, format='gps', scale='utc')
        obs_source.obstime = t
        obs_source_altaz = obs_source.transform_to('altaz')

        if obs_source_altaz.alt.deg < 15.0:
            logger.debug("Source below pointing horizon at this time, skip this timestep.")
            continue  # Source below pointing horizon at this time, skip this timestep.

        avoid_source.obstime = t
        avoid_source_altaz = avoid_source.transform_to('altaz')  # Use as, for example: avoid_source_altaz.az.deg

        if avoid_source_altaz.alt.deg < 0.0:
            tracklist.append((starttime, step, obs_source_altaz.az.deg, obs_source_altaz.alt.deg))
            logger.debug("Avoided source below TRUE horizon, just use actual target az/alt for this timestep.")
            continue  # Avoided source below TRUE horizon, just use actual target az/alt for this timestep.

        dist_deg = obs_source.separation(avoid_source).deg

        logger.debug("Observed source at (az,alt) = (%.4f,%.4f) [deg]" % (obs_source_altaz.az.deg, obs_source_altaz.alt.deg))
        logger.debug("Avoided  source at (az,alt) = (%.4f,%.4f) [deg]" % (avoid_source_altaz.az.deg, avoid_source_altaz.alt.deg))
        logger.debug("Anglular distance = %.2f [deg]" % (dist_deg))
        logger.debug("Gps time = %d" % t.gps)

        gp_positions = astropy.coordinates.SkyCoord(alt=gp_alts,
                                                    az=gp_azes,
                                                    unit=astropy.units.deg,
                                                    location=config.MWAPOS,
                                                    frame=astropy.coordinates.AltAz,
                                                    obstime=t)

        dist_obs_degs = obs_source_altaz.separation(gp_positions).deg
        dist_avoid_degs = avoid_source_altaz.separation(gp_positions).deg

        # select gridpoints within given angular distance :
        best_gridpoint = None
        r_max = -1000
        best_gain_obs = 0
        best_gain_avoid = 0
        skipped_too_far = 0
        skipped_gain_too_low = 0
        for i in range(len(gp_numbers)):
            gpnum = gp_numbers[i]
            dist_obs = dist_obs_degs[i]
            dist_avoid = dist_avoid_degs[i]

            if verb_level > 1:
                outstring = "\n\t\ttesting gridpoint %d, dist_obs_deg = %.2f [deg], dist_avoid_deg = %.2f [deg]"
                logger.debug(outstring % (gpnum, dist_obs, dist_avoid))

            # if dist_obs_deg < options.max_beam_distance_deg and dist_avoid_deg < options.max_beam_distance_deg :
            if dist_obs < max_beam_distance_deg:
                beam_obs = primarybeammap_tant.get_beam_power(t.gps,
                                                              gp_delays[i],
                                                              freq,
                                                              model=model,
                                                              pointing_az_deg=obs_source_altaz.az.deg,
                                                              pointing_za_deg=90 - obs_source_altaz.alt.deg,
                                                              zenithnorm=True)
                beam_avoid = primarybeammap_tant.get_beam_power(t.gps,
                                                                gp_delays[i],
                                                                freq,
                                                                model=model,
                                                                pointing_az_deg=avoid_source_altaz.az.deg,
                                                                pointing_za_deg=90 - avoid_source_altaz.alt.deg,
                                                                zenithnorm=True)

                gain_XX_obs = beam_obs['XX']
                gain_XX_avoid = beam_avoid['XX']
                r = gain_XX_obs / gain_XX_avoid

                if r > 1.00 and gain_XX_obs > min_gain:
                    outstring = "\t\tSelected gridpoint = %d at (az,elev) = (%.4f,%.4f) [deg] at (distances %.4f and %.4f deg) "
                    outstring += "-> gain_obs=%.4f and gain_avoid=%.4f -> gain_obs/gain_avoid = %.4f"
                    logger.debug(outstring % (gpnum, gp_azes[i], gp_alts[i], dist_obs, dist_avoid, gain_XX_obs, gain_XX_avoid, r))
                    if r > r_max:
                        best_gridpoint = i
                        r_max = r
                        best_gain_obs = gain_XX_obs
                        best_gain_avoid = gain_XX_avoid
                else:
                    skipped_gain_too_low = skipped_gain_too_low + 1
                    if verb_level > 1:
                        outstring = "\t\tSKIPPED gridpoint = %d at (az,elev) = (%.4f,%.4f) [deg] at (distances %.4f and %.4f deg) "
                        outstring += "-> gain_obs=%.4f (vs. min_gain=%.2f) and gain_avoid=%.4f -> gain_obs/gain_avoid = %.4f"
                        logger.debug(outstring % (gpnum,
                                                  gp_azes[i],
                                                  gp_alts[i],
                                                  dist_obs,
                                                  dist_avoid,
                                                  gain_XX_obs,
                                                  min_gain,
                                                  gain_XX_avoid, r))
            else:
                skipped_too_far = skipped_too_far + 1
                if verb_level > 1:
                    outstring = "\t\t\tskipped as dist_obs_deg = %.2f [deg] and dist_avoid_deg = %.2f [deg] , one >  "
                    outstring += "max_beam_distance_deg = %.2f [deg]"
                    logger.debug(outstring % (dist_obs, dist_avoid, max_beam_distance_deg))

        logger.debug("Number of gridpoints skipped due to gain lower than minimum (=%.2f) = %d" % (min_gain,
                                                                                                   skipped_gain_too_low))
        outstring = "Number of gridpoints skipped due to being further than limit ( max_beam_distance_deg = %.2f [deg] ) = %d"
        logger.debug(outstring % (max_beam_distance_deg, skipped_too_far))

        if best_gridpoint is not None:
            outstring = "Best gridpoint %d at (az,alt)=(%.4f,%.4f) [deg] at %s UTC to observe has ratio = %.2f = %.8f / %.8f\n"
            logger.debug(outstring % (gp_numbers[best_gridpoint],
                                      gp_azes[best_gridpoint],
                                      gp_alts[best_gridpoint],
                                      t.iso, r_max,
                                      best_gain_obs,
                                      best_gain_avoid))
            tracklist.append((starttime, step, gp_azes[best_gridpoint], gp_alts[best_gridpoint]))

    return tracklist


def get_best_gridpoints_supress_sun(gps_start,
                                    obs_source_ra_deg,
                                    obs_source_dec_deg,
                                    model="analytic",
                                    min_gain=0.5,
                                    max_beam_distance_deg=30,
                                    channel=145,
                                    verb_level=1,
                                    duration=3600,
                                    step=120):
    t = Time(gps_start, format='gps', scale='utc')
    sunpos = get_sun(t)
    return get_best_gridpoints(gps_start=gps_start,
                               obs_source_ra_deg=obs_source_ra_deg,
                               obs_source_dec_deg=obs_source_dec_deg,
                               avoid_source_ra_deg=sunpos.ra.deg,
                               avoid_source_dec_deg=sunpos.dec.deg,
                               model=model,
                               min_gain=min_gain,
                               max_beam_distance_deg=max_beam_distance_deg,
                               channel=channel,
                               verb_level=verb_level,
                               duration=duration,
                               step=step)


def get_sun_elevation(gps_start=None):
    if gps_start is None:
        t = Time.now()
    else:
        t = Time(gps_start, format='gps', scale='utc')
    sunpos = astropy.coordinates.get_sun(t)
    sunpos.location = config.MWAPOS
    sunprec = sunpos.transform_to('altaz')

    return sunprec.alt.deg
