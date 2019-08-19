
import os

import skyfield.api as si

import astropy
import astropy.time

MWA_TOPO = si.Topos(longitude=(116, 40, 14.93), latitude=(-26, 42, 11.95), elevation_m=377.8)
PLANETS = None
TIMESCALE = None
S_MWAPOS = None


def init_data():
    global PLANETS, TIMESCALE, S_MWAPOS

    if (PLANETS is None) or (TIMESCALE is None) or (S_MWAPOS is None):
        if 'XDG_CACHE_HOME' in os.environ:
            datadir = os.environ['XDG_CACHE_HOME'] + '/skyfield'
        else:
            datadir = '/tmp'
        skyfield_loader = si.Loader(datadir, verbose=False, expire=False)
        PLANETS = skyfield_loader('de421.bsp')
        TIMESCALE = skyfield_loader.timescale()
        S_MWAPOS = PLANETS['earth'] + MWA_TOPO


def time2tai(input_time=None):
    """Converts an arbitrary input time into a skyfield.api.Time object.
       If the input is in GPS seconds, or an an Astropy.time.Time object, it is converted
       appropriately. If it's already a skyfield.api.Time object, it's returned as-is.
       If None is passed, the current time is returned

       :param input_time: time to convert
       :return: a skyfield.api.Time object
    """
    init_data()
    if input_time is None:
        return TIMESCALE.now()
    elif type(input_time) is si.Time:
        return input_time
    elif type(input_time) in [int, float, long]:  # Must be in GPS seconds
        # Offset at Jan 6, 1980 is 19 seconds.
        return TIMESCALE.tai(jd=2444244.5 + (input_time + 19) / 86400.0)
    elif type(input_time) is astropy.time.Time:
        return TIMESCALE.from_astropy(input_time)
    else:
        return None


def tai2gps(tai=None):
    init_data()
    if tai is None:
        tai = TIMESCALE.now()
    # Offset at Jan 6, 1980 is 19 seconds.
    return (tai.tai - 2444244.5) * 86400 - 19


"""
Analysis of alt/az differences for Sirius, assuming RA=06:45.136 (101.2840), Dec=-16:43.367 (-16.72278333)
Reference Time: 2019/08/15T00:19:20.0000 = 1249863578.0000
            
Skyfield:
 at ..78.0000   Alt=71:27:48.86371 (71.4635732538), Az=60:55:42.9285432 (60.92859126200)

Astropy:        
 at ..78.0000   Alt=71:27:52.4400 (71.46456668), Az=60:55:33.3089 (60.92591915)
 at ..77.69547  Alt=71:27:48.8637 (71.46357325), Az=60:55:41.2975 (60.928138194)  # best alt match
 at ..77.6333   Alt=71:27:48.1336 (71.46337044), Az=60:55:42.9283 (60.928591194)  # best az match
 at ..77.664385 Alt=71:27:48.4986 (71.46347183), Az=60:55:42.1129 (60.928364694) # Best overall match

Differences are 3.6/9.6 arcsec respectively for the same time, or 0.4/0.8 arcsec if a 0.335615 sec time offset is applied.

Presumably differences are due to differences in the latitude/longitude datum and Earth shape model (WGS84, etc)

"""