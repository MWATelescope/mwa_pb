
import os

import skyfield.api as si

import astropy
import astropy.time

# Standard MWA lat/long are in WGS84 datum, but skyfield doesn't use WGS84. Makes for ~10 arcsec differences in alt/azes.
MWA_TOPO = si.Topos(longitude=(116, 40, 14.93), latitude=(-26, 42, 11.95), elevation_m=377.8)

PLANETS = None
TIMESCALE = None
S_MWAPOS = None

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

Over the whole sky, the entire MWA sweetspot list in alt/az was converted to RA/Dec at the MWA location, and
the same reference time as above. The differences between astropy and skyfield, in arcseconds:

For RA:
[16.7, 16.1, 18.9, 17.5, 14.2, 18.1, 20.1, 14.6, 13.8, 15.7, 20.9, 18.7, 11.5, 17.6, 19.9, 22.2, 21.8, 15.3, 
11.5, 11.4, 13.6, 19.3, 24.3, 11.5, 11.4, 15.5, 22.5, 20.5, 8.7, 17.3, 21.4, 23.9, 24.3, 16.1, 8.3, 8.9, 13.5, 
18.9, 20.8, 26.1, 27.3, 11.2, 7.6, 9.0, 11.4, 15.4, 23.7, 23.3, 5.7, 17.2, 22.6, 25.1, 28.5, 17.1, 4.9, 6.2, 
13.5, 20.4, 29.2, 6.3, 9.1, 18.9, 22.0, 27.2, 32.0, 10.3, 3.7, 6.5, 11.4, 15.6, 20.4, 21.7, 24.5, 30.0, 33.8, 
28.3, 3.9, 1.6, 2.7, 6.6, 9.1, 17.4, 23.5, 25.8, 36.1, 18.1, 1.6, 3.4, 13.6, 19.1, 23.0, 27.5, 40.3, 7.6, 0.0, 
3.8, 11.4, 21.7, 33.8, -1.8, 6.6, 20.7, 22.7, 29.8, 40.9, -1.5, -2.6, 4.0, 9.1, 16.0, 24.8, 39.4, -0.3, 18.0, 
24.0, 25.8, 54.0, 17.1, -1.6, 0.4, 13.8, 19.9, 23.6, 27.1, 56.4, -2.4, -3.6, 0.9, 11.5, 22.2, 22.9, 32.4, 38.7, 
-8.4, -6.7, 3.9, 6.4, 21.7, 23.5, 28.5, 50.8, -14.5, -6.4, 1.0, 8.9, 17.0, 24.5, 91.1, -3.6, 19.4, 23.6, 24.1, 
25.1, 34.5, 119.7, -21.1, -13.2, -5.0, -2.7, 3.4, 14.4, 23.6, 23.9, 29.5, 42.1, -20.2, -10.6, 0.6, 5.9, 21.7, 
23.9, 25.7, 79.2, -43.8, -7.0, -2.3, 11.6, 24.2, 24.0, 26.0, 51.0, -39.9, -9.9, -2.5, 8.3, 19.7, 23.2, -67.3, 
-7.4, 23.5, 23.2, 23.1, 10.2, -84.7, -8.7, -6.8, 16.0]

For Dec:
[-13.2, -11.7, -12.4, -14.4, -13.7, -11.2, -13.4, -15.1, -12.1, -10.1, -11.5, -15.5, -14.0, -9.8, -10.5, -12.3, 
-14.2, -16.4, -15.5, -12.3, -10.3, -9.3, -12.7, -16.9, -10.4, -8.3, -10.5, -16.3, -14.0, -8.1, -9.7, -11.0, -14.7, 
-17.5, -15.6, -12.2, -8.4, -7.8, -8.7, -11.1, -12.8, -18.1, -17.0, -10.2, -8.3, -6.3, -9.3, -17.0, -13.7, -6.3, 
-8.8, -9.5, -14.9, -18.4, -15.3, -11.9, -6.2, -7.5, -10.7, -18.2, -8.0, -6.2, -8.1, -9.3, -12.4, -19.2, -16.7, 
-9.8, -6.0, -4.0, -6.0, -7.1, -8.2, -8.5, -9.6, -17.3, -19.2, -17.8, -13.1, -7.5, -5.6, -4.2, -8.0, -8.1, -14.4, 
-19.2, -14.6, -11.2, -3.7, -4.3, -7.4, -7.5, -10.9, -20.0, -15.9, -9.2, -3.3, -5.9, -6.8, -18.4, -5.0, -4.3, -6.7, 
-6.3, -7.2, -19.6, -16.8, -6.8, -2.8, -1.3, -7.1, -17.3, -12.1, -1.7, -7.1, -6.6, -12.5, -20.0, -13.6, -10.3, -0.9, 
-2.1, -6.9, -5.7, -7.1, -20.2, -14.7, -8.2, -0.3, -4.5, -5.8, -4.2, -3.7, -18.2, -17.0, -4.1, -2.0, -2.4, -6.4, 
-4.2, -2.3, -18.6, -15.2, -5.8, 0.4, 1.7, -6.1, -16.7, -10.6, 1.1, -4.7, -6.4, -5.4, -0.7, -4.4, -19.6, -16.1, 
-12.0, -8.9, -1.0, 2.5, -2.9, -5.9, -1.8, 1.5, -16.0, -14.9, -3.0, 1.4, 0.4, -6.5, -4.2, 3.2, -16.2, -12.8, -6.8, 
3.4, -0.3, -6.5, -2.4, 7.1, -12.5, -12.9, -4.3, 4.6, 6.2, -5.6, 15.2, -8.5, 5.0, -6.3, -4.6, 17.6, 5.6, -9.5, -6.8, 
7.5]

"""


def init_data():
    global PLANETS, TIMESCALE, S_MWAPOS

    if (PLANETS is None) or (TIMESCALE is None) or (S_MWAPOS is None):
        if 'XDG_CACHE_HOME' in os.environ:
            datadir = os.environ['XDG_CACHE_HOME'] + '/skyfield'
        else:
            datadir = '/tmp'
        skyfield_loader = si.Loader(datadir, verbose=False, expire=True)
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
    """
    Converts a skyfield time object into a value in GPS seconds.

    :param tai: skyfield time object
    :return: value in GPS seconds.
    """
    init_data()
    if tai is None:
        tai = TIMESCALE.now()
    # Offset at Jan 6, 1980 is 19 seconds.
    return (tai.tai - 2444244.5) * 86400 - 19
