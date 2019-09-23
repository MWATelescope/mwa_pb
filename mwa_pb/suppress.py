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

import skyfield.api as si

import primarybeammap_tant
import mwa_sweet_spots
import skyfield_utils as su


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
                        step=120,
                        min_elevation=50.00):
    su.init_data()
    frequency = channel * 1.28

    if model not in ['analytic', 'advanced', 'full_EE', 'full_EE_AAVS05']:
        logger.error("Model %s not found\n" % model)

    gp_numbers = list(mwa_sweet_spots.all_grid_points.keys())
    gp_numbers.sort()
    gp_azes = numpy.array([mwa_sweet_spots.all_grid_points[i][1] for i in gp_numbers])
    gp_alts = numpy.array([mwa_sweet_spots.all_grid_points[i][2] for i in gp_numbers])
    gp_delays = [mwa_sweet_spots.all_grid_points[i][4] for i in gp_numbers]

    obs_source = si.Star(ra=si.Angle(degrees=obs_source_ra_deg),
                         dec=si.Angle(degrees=obs_source_dec_deg))

    avoid_source = si.Star(ra=si.Angle(degrees=avoid_source_ra_deg),
                           dec=si.Angle(degrees=avoid_source_dec_deg))

    freq = frequency * 1e6
    tracklist = []  # List of (starttime, duration, az, el) tuples
    for starttime in range(int(gps_start), int(gps_start + duration), int(step)):
        t = su.time2tai(starttime)
        observer = su.S_MWAPOS.at(t)
        obs_source_apparent = observer.observe(obs_source).apparent()
        obs_source_alt, obs_source_az, _ = obs_source_apparent.altaz()

        if obs_source_alt.degrees < min_elevation:
            logger.debug("Source at %.2f [deg] below minimum elevation = %.2f [deg]  at this time, skip this timestep." % (obs_source_alt.degrees,
                                                                                                                           min_elevation))
            continue  # Source below pointing horizon at this time, skip this timestep.

        avoid_source_apparent = observer.observe(avoid_source).apparent()
        avoid_source_alt, avoid_source_az, _ = avoid_source_apparent.altaz()

        if avoid_source_alt.degrees < 0.0:
            tracklist.append((starttime, step, obs_source_az.degrees, obs_source_alt.degrees))
            logger.debug("Avoided source below TRUE horizon, just use actual target az/alt for this timestep.")
            continue  # Avoided source below TRUE horizon, just use actual target az/alt for this timestep.

        dist_deg = obs_source_apparent.separation_from(avoid_source_apparent).degrees

        logger.debug("Observed source at (az,alt) = (%.4f,%.4f) [deg]" % (obs_source_az.degrees, obs_source_alt.degrees))
        logger.debug("Avoided  source at (az,alt) = (%.4f,%.4f) [deg]" % (avoid_source_az.degrees, avoid_source_alt.degrees))
        logger.debug("Anglular distance = %.2f [deg]" % (dist_deg))
        logger.debug("Gps time = %d" % su.tai2gps(t))

        gp_positions = observer.from_altaz(alt_degrees=gp_alts,
                                           az_degrees=gp_azes,
                                           distance=si.Distance(au=9e90))

        dist_obs_degs = obs_source_apparent.separation_from(gp_positions).degrees
        dist_avoid_degs = avoid_source_apparent.separation_from(gp_positions).degrees

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
                beam_obs = primarybeammap_tant.get_beam_power(gp_delays[i],
                                                              freq,
                                                              model=model,
                                                              pointing_az_deg=obs_source_az.degrees,
                                                              pointing_za_deg=90 - obs_source_alt.degrees,
                                                              zenithnorm=True)
                beam_avoid = primarybeammap_tant.get_beam_power(gp_delays[i],
                                                                freq,
                                                                model=model,
                                                                pointing_az_deg=avoid_source_az.degrees,
                                                                pointing_za_deg=90 - avoid_source_alt.degrees,
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
                                      t.utc_iso(), r_max,
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
                                    step=120,
                                    min_elevation=50.00):
    t = su.time2tai(gps_start)
    sunra, sundec, _ = su.S_MWAPOS.at(t).observe(su.PLANETS['Sun']).apparent().radec()
    return get_best_gridpoints(gps_start=gps_start,
                               obs_source_ra_deg=obs_source_ra_deg,
                               obs_source_dec_deg=obs_source_dec_deg,
                               avoid_source_ra_deg=sunra.hours * 15.0,
                               avoid_source_dec_deg=sundec.degrees,
                               model=model,
                               min_gain=min_gain,
                               max_beam_distance_deg=max_beam_distance_deg,
                               channel=channel,
                               verb_level=verb_level,
                               duration=duration,
                               step=step,
                               min_elevation=min_elevation)


def get_sun_elevation(gps_start=None):
    t = su.time2tai(gps_start)
    sunalt, sunaz, _ = su.S_MWAPOS.at(t).observe(su.PLANETS['Sun']).apparent().altaz()

    return sunalt.degrees
