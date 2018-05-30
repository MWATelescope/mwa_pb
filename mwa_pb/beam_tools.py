#!/usr/bin/env python
"""Tools for transforming and plotting a primary beam generated with beam_full_EE.py.
   Tim Colegate, updated from Randall Wayth's mwa_tile.py.
"""
import numpy as np
import math

import logging

import matplotlib.pyplot as plt
import scipy.io as io

logger = logging.getLogger('beam_tools')


def makeAZZA(npix=256, projection='SIN'):
    """
       Make azimuth and zenith angle arrays for a square image of side npix
       Projection is SIN or ZEA, all-sky
       Returns (az,za). Angles are in radian. ZA values are nan beyond the horizon
    """
    # build az and za arrays
    # use linspace to ensure we go to horizon on all sides
    z = np.linspace(-npix / 2.0, npix / 2.0, num=npix, dtype=np.float32)
    x = np.empty((npix, npix), dtype=np.float32)
    y = np.empty((npix, npix), dtype=np.float32)
    for i in range(npix):
        y[i, 0:] = z
        x[0:, i] = z
    d = np.sqrt(x * x + y * y) / (npix / 2)
    # only select pixels above horizon
    t = (d <= 1.0)
    za = np.zeros((npix, npix), dtype=np.float32) * np.NaN
    if projection == 'SIN':
        za[t] = np.arcsin(d[t])
        logger.info('Using slant orthographic projection')
    elif projection == 'ZEA':
        d = d * 2 ** 0.5  # ZEA requires R to extend beyond 1.
        za[t] = 2 * np.arcsin(d[t] / 2.0)
        logger.info('Using zenithal equal area projection')
    else:
        e = 'Projection %s not found' % projection
        logger.error(e)
        raise ValueError(e)
    az = np.arctan2(y, x)
    az = az + math.pi  # 0 to 2pi
    az = 2 * math.pi - az  # Change to clockwise from top (when origin is in top-left)

    return az, za


def makeAZZA_dOMEGA(npix=256, projection='SIN'):
    """
       Make azimuth and zenith angle arrays for a square image of side npix
       Projection is SIN or ZEA, all-sky
       Returns (az,za). Angles are in radian. ZA values are nan beyond the horizon
       This one also returns number of pixels and dOMEGA map which is used in sensitivity calculation in mwa_sensitivity.py script
    """
    # build az and za arrays
    # use linspace to ensure we go to horizon on all sides
    z = np.linspace(-npix / 2.0, npix / 2.0, num=npix, dtype=np.float32)
    x = np.empty((npix, npix), dtype=np.float32)
    y = np.empty((npix, npix), dtype=np.float32)
    dOMEGA = np.empty((npix, npix), dtype=np.float32)

    for i in range(npix):
        y[i, 0:] = z
        x[0:, i] = z
    d = np.sqrt(x * x + y * y) / (npix / 2)
    # only select pixels above horizon
    t = (d <= 1.0)
    n_total = t.sum()
    dOMEGA.fill(math.pi * 2.00 / n_total)

    za = np.zeros((npix, npix), dtype=np.float32) * np.NaN
    if projection == 'SIN':
        za[t] = np.arcsin(d[t])
        dOMEGA = np.cos(za) * math.pi * 2.00 / n_total
        logger.info('Using slant orthographic projection')

    elif projection == 'ZEA':
        d = d * 2 ** 0.5  # ZEA requires R to extend beyond 1.
        za[t] = 2 * np.arcsin(d[t] / 2.0)
        logger.info('Using zenithal equal area projection')
    else:
        e = 'Projection %s not found' % projection
        logger.error(e)
        raise ValueError(e)
    az = np.arctan2(y, x)
    az = az + math.pi  # 0 to 2pi
    az = 2 * math.pi - az  # Change to clockwise from top (when origin is in top-left)

    return (az, za, n_total, dOMEGA)


def makeUnpolInstrumentalResponse(j1, j2):
    # TODO: check this description below. I think Jones dimensions are now swapped
    """
       Form the visibility matrix in instrumental response from two Jones
       matrices assuming unpolarised sources (hence the brightness matrix is
       the identity matrix)
       Input: j1,j2: Jones matrices of dimension[za][az][2][2]
       Returns: [za][az][[xx,xy],[yx,yy]] where "X" and "Y" are defined by the receptors
       of the Dipole object used in the ApertureArray. Hence to get "XX", you want
       result[za][az][0][0] and for "YY" you want result[za][az][1][1]
    """
    result = np.empty_like(j1)

    result[0, 0] = j1[0, 0] * j2[0, 0].conjugate() + j1[0, 1] * j2[0, 1].conjugate()
    result[1, 1] = j1[1, 0] * j2[1, 0].conjugate() + j1[1, 1] * j2[1, 1].conjugate()
    result[0, 1] = j1[0, 0] * j2[1, 0].conjugate() + j1[0, 1] * j2[1, 1].conjugate()
    result[1, 0] = j1[1, 0] * j2[0, 0].conjugate() + j1[1, 1] * j2[0, 1].conjugate()
    return result


# def makePolInstrumentResponse(j1, j2, b):
#   """
#      Form the instrument response from two Jones matrices with an
#      arbitrary source brightness matrix, hence arbitrary polarisation
#      Returns: (xx,yy,xy,yx) where "X" and "Y" are defined by the receptors
#      of the Dipole object used in the ApertureArray
#   """
#   # FIXME: need to work out how to do this in vectorised way.
#   pass


def plotArrayJones(j, freq, filebase, title, pix_per_deg=1, j_1D=None, gridded=False):
    """
       Utility to plot the output of tile Jones matrices
       Input:
       j_1D - 1-D cut along an azimuth angle
    """
    plt.rcParams['savefig.dpi'] = 300

    for i in [0, 1]:
        for ii in [0, 1]:
            if j_1D is not None:  # show cut
                plt.subplot(121)
                plt.plot(np.arange(len(j_1D[i, ii])) * 1.0 / pix_per_deg, np.abs(j_1D[i, ii]))
                plt.title('1-D cut')
                plt.xlabel('ZA (degs)')
                plt.ylabel('magnitude')
                plt.subplot(122)

            if gridded:
                plt.imshow(np.abs(j[i, ii]), interpolation='none', extent=[0, 90, 360, 0])
                plt.xticks(np.arange(0, 91, 30))
                plt.yticks(np.arange(360, -1, -30))
            else:
                plt.imshow(np.abs(j[i, ii]), interpolation='none')
            plt.suptitle('MWA %s MHz J%s%s voltage mag, %s' % (freq / 1.e6, i, ii, title))
            plt.colorbar(label='magnitude')
            #    plt.gca().invert_yaxis()
            plt.savefig('MWA_J%s%s_voltage_mag_%sMHz_%s.png' % (i, ii, freq / 1.e6, filebase))
            plt.clf()

            if j_1D is not None:  # show cut
                plt.subplot(121)
                plt.plot(np.arange(len(j_1D[i, ii])) * 1.0 / pix_per_deg, np.angle(j_1D[i, ii]) * 180 / math.pi)
                plt.title('1-D cut')
                plt.xlabel('ZA (deg)')
                plt.ylabel('phase (deg)')
                plt.subplot(122)

            if gridded:
                plt.imshow(np.angle(j[i, ii]) * 180 / math.pi, interpolation='none', extent=[0, 90, 360, 0])
                plt.xticks(np.arange(0, 91, 30))
                plt.yticks(np.arange(360, -1, -30))
            else:
                plt.imshow(np.angle(j[i, ii]) * 180 / math.pi, interpolation='none')
            plt.suptitle('MWA %s MHz J%s%s voltage phase, %s' % (freq / 1e6, i, ii, title))
            plt.colorbar(label='phase (deg)')
            #    plt.gca().invert_yaxis()
            plt.savefig('MWA_J%s%s_voltage_phase_%sMHz_%s.png' % (i, ii, freq / 1.e6, filebase))
            plt.clf()


def exportArrayJones(j, freq, filebase):
    """
       Utility to export the output of tile Jones matrices to a .mat file
    """
    filename = 'MWA_voltage_' + str(freq / 1e6) + 'MHz' + filebase + '.mat'
    mydata = {}

    for i in [0, 1]:
        for ii in [0, 1]:
            mydata['J%s%s' % (i, ii)] = j[i, ii]
    io.savemat(filename, mydata)


def plotVisResponse(j, freq, filebase, title, pix_per_deg, gridded=False):
    # TODO: check this description below. I think Jones dimensions are now swapped
    """
       Utility to plot the visibilty XX,YY,XY and YX response of the array for
       an unpolarised 1Jy source
       Input: j a visibility matrix (complex) of dimensions [za][az][2][2]
    """
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['axes.titlesize'] = 'medium'

    vis = makeUnpolInstrumentalResponse(j, j)
    vis = np.abs(vis)
    data = {'XX': vis[0, 0], 'YY': vis[1, 1],
            'XY': vis[0, 1], 'YX': vis[1, 0]}
    for key, val in data.iteritems():
        my_max = np.max(val, axis=1)  # Max za for each az
        max_idx = np.argmax(my_max)  # Find az index
        if gridded:  # show cut
            plt.subplot(121)
            plt.plot(np.arange(len(j[0, 0, 0, :])) * 1.0 / pix_per_deg, val[max_idx, :])
            plt.title('Cut at Az=%.1f' % ((max_idx + 1) * 1 / pix_per_deg))
            plt.xlabel('ZA (degs)')
            plt.ylabel('power')
            plt.subplot(122)
            plt.imshow(val, interpolation='none', extent=[0, 90, 360, 0])
            plt.xticks(np.arange(0, 91, 30))
            plt.yticks(np.arange(360, -1, -30))
        else:
            plt.imshow(val, interpolation='none')
        plt.colorbar(label='power')
        # plt.gca().invert_yaxis()

        plt.suptitle('MWA ' + str(freq / 1.e6) + 'MHz ' + key + ' mag ' + title)
        plt.savefig('MWA_' + key + '_mag_' + str(freq / 1.e6) + 'MHz_' + filebase + '.png')
        plt.clf()
