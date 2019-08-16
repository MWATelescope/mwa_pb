
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
Time: 2019/08/15T01:26:00 = 1249867578

Starry Night:   RA=06:45.136 (101.2840), Dec=-16:43.367 (-16.72278333)
                Alt=80:2.593 (80.0432167), Az=0:51.046 (0.85076667)

Astropy:        RA=06:45:8.9172792 (06:45.14862, 101.28715533), Dec=-16:42:58.017096 (-16:42.9669516, -16.71611586)
                Alt=80:01:51.1472899 (80:01.852455, 80.030874247), Az=0:51:48.8061285 (0:51.813435475, 0.8635572579)
at SN locn:     Alt=80:02:15.2776712 (80:02.2546279, 80.037577131), Az=0:50:47.612721 (0:50.793545344, 0.8465590891)
                
Skyfield:       RA=06:45:8.9172792 (06:45.14862, 101.28715533), Dec=-16:42:58.017096 (-16:42.9669516, -16.71611586)
                Alt=80:01:51.32362 (80:01.85539367, 80.0309232279), Az=00:52:28.9320034 (0:52.482200057, 0.87470333428)
at SN locn:     Alt=80:02:15.45593 (80:02.25759882, 80.0376266470), Az=00:51:27.7636801 (0:51.462728001, 0.857712133352)

"""