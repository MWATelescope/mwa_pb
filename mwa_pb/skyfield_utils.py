
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


# def tai2gps(tai=None):
#     config.init_data()
#     if tai is None:
#         tai = config.TIMESCALE.now()
#     # Offset at Jan 6, 1980 is 19 seconds.
#     return (tai.tai - 2444244.5) * 86400 - 19
