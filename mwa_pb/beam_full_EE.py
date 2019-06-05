#!/usr/bin/env python
"""beam_full_EE.py
Determine complex voltage response of an aperture array modelled as embedded elements,  
represented by spherical harmonics. 

Usage:
An ApertureArray object is created for a target frequency.
A Beam object takes an ApertureArray object and antenna delays and amplitude 
as input, calculates the beam response as the weighted sum of 
the spherical harmonics of the embedded elements.
The Beam object can then be queried to return the response at 
spherical coordinates or interest, to an arbitrary angular resolution.
The response can either be calculated at the given points, or for faster 
processing, interpolated from a gridded beam.

A gridded beam is significantly faster for a large number of points, 
as the time-consuming calculation of the response 
need only be done once for each theta (ZA) and each phi (az) angle. 
The calculation for each (theta,phi) point is then a simple multiplication.
A linear interpolation from the gridded beam is required as the 
slant orthographic projection results in many unique theta (ZA) and phi (az) points, 
but this is also relatively fast.

If this module is run stand-alone, 
the a beam will be generated and the various outputs plotted.

Original Matlab code by Adrian Sutinjo. Ported to Python by Tim Colegate. 
See:
Calculating Far-Field Radiation Based on FEKO
 Spherical Wave Coefficients, draft 10 June 2015
 
"""

import datetime  # For info-level logging
import numpy as np
import logging
import os
import math

from scipy import misc
from scipy.special import lpmv  # associated Legendre function

from astropy.io import fits
from scipy.interpolate import RegularGridInterpolator

from . import config
from . import beam_tools

logging.basicConfig(format='# %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)  # default logger level is WARNING
# logger.setLevel('WARNING')

try:
    import h5py
except ImportError:
    h5py = None
    logger.warning("Cannot import h5py module -> Full Embedded Model cannot be used, but other beam models should work fine")

deg2rad = math.pi / 180
rad2deg = 180 / math.pi


# Todo: check scipy.__version__ >= '0.15.1'


class ApertureArray(object):
    """Generic aperture array class"""

    # TODO: add some checks to h5 file. e.g. check n_ant matches number in h5 file.
    # TODO: in the beam modelling, there is still 16 antenna hardcoded. This needs fixing.
    def __init__(self, h5filepath, target_freq_Hz, n_ant=16):
        # freq_interp=nearest, linear
        """Constructor for beamformed aperture array.
           Input:
           h5filepath - path to h5 file containing coefficients
           target_freq_Hz - frequency at which beam model is required
           n_ant - number of antennas in array/tile
        """
        logger.info('New model of the physical tile, modelled using full embedded element patterns with the beam described by spherical harmonics.')
        logger.info('This new beam model is still being tested and is not an official release')
        logger.info('Code version date: 2017-07-20 (Sigma_P sign flipped standalone version)')

        # load h5file
        if not os.path.exists(h5filepath):
            logger.error('Cannot file beam model file %s' % h5filepath)
            self.h5f = None
        else:
            logger.debug('Loading beam model from file %s' % h5filepath)
            self.h5f = h5py.File(h5filepath, 'r')

        # read and log version information :
        self.h5_file_version = config.h5fileversion
        logger.debug("H5 file (%s) version = %s" % (h5filepath, self.h5_file_version))

        # Find available frequencies in h5 file
        freqs = np.array([int(x[3:]) for x in list(self.h5f.keys()) if 'X1_' in x])
        freqs.sort()

        # find the nearest freq lookup table
        pos = np.argmin(np.abs(freqs - target_freq_Hz))
        self.freq = freqs[pos]
        logger.info("%s MHz requested, selecting nearest freq: %s MHz" % (target_freq_Hz / 1.e6, self.freq / 1.e6))

        self.n_ant = n_ant
        self.norm_fac = None

    def calc_zenith_norm_fac(self):
        """Calculate normalisation factors for the Jones vector for this
        ApertureArray object. For MWA, these are at the zenith of a zenith pointed beam,
        which is the maximum for all beam pointings.
        The FEKO simulations include all ph angles at za=0. These are not redundant,
        and the ph value determines the unit vector directions of both axes.
        For the E-W dipoles, the projection of the theta unit vec will be max when
        pointing east, i.e. when ph_EtN=0 (ph_NtE=90). For the phi unit vec,
        this will be when ph_EtN=-90 or 90 (ph_NtE=180 or 0: we use 180)
        For the N-S dipoles, projection of ZA onto N-S is max az ph_EtN=90 (ph_NtE=0) and
        proj of ph onto N-S is max when ph_EtN=0 (ph_NtE=90)"""

        mybeam = Beam(self, delays=np.zeros([2, 16]), amps=np.ones([2, 16]))
        self.norm_fac = np.zeros((2, 2), dtype=np.complex128)

        # fill in Jones matrix
        # 2017-05-31 : MS changed normalisation at zenith to ABS so that normalisation does not change signs of the Jones matrix
        #              otherwise signs might flip again, at the moment they are as the Jones matrix is calculated and normalisation does not mess up with it
        #              It probably needs some proper normalisation which Adrian and Daniel and a bit MS are still working on ...
        max_phis = [[math.pi / 2, math.pi], [0, math.pi / 2]]  # phi where each Jones vector is max
        for i in [0, 1]:
            for ii in [0, 1]:
                self.norm_fac[i][ii] = abs(mybeam.get_response(max_phis[i][ii], 0)[i][ii][0])  # or with abs

        # print "----------------------------------------------------------------"
        # print "Normalisation Jones matrix :"
        # print "----------------------------------------------------------------"
        # print "   %s     |     %s " % (self.norm_fac[0][0],self.norm_fac[0][1])
        # print "   %s     |     %s " % (self.norm_fac[1][0],self.norm_fac[1][1])
        # print "----------------------------------------------------------------"

    def apply_zenith_norm_Jones(self, j):
        """Apply zenith normalisation factor to the Jones matrix

           Input:
             j - Jones matrix for one or more spherical cordinates
        """
        if not self.norm_fac:
            self.calc_zenith_norm_fac()
        # Resize to extra dimensions for subsequent broadcasting during normalisation
        mynorm_fac = np.copy(self.norm_fac)
        for i in range(len(j.shape) - 2):
            mynorm_fac = np.expand_dims(mynorm_fac, axis=2)
        return j / mynorm_fac  # Normalise
        # return j    # normalisation turned off for tests


class Beam(object):
    def __init__(self, AA, delays=None, amps=None):
        """Constructor for aperture array beam of given pointing direction
        and angular resolution.
        Spherical harmonics modes and coefficients are accumulated for
        a pointing defined by delays and amps.

        The format of the modes and coefficients is as follows:
        Q1,2 is Q1,2mn column vector with m=[-n:n].' and n=[111]....[n,..n].'
        M and N assume FEKO M and N vector order, e.g,...:
        -----------M, N------------
             -1     1
              0     1
              1     1
             -2     2
             -1     2
              0     2
              1     2
              2     2
              ..   ..
              -nmax nmax
              ..     ..
              0     nmax
              ..    ..
              nmax  nmax

        Input:
        AA - aperture array object antenna object
        freq - frequency of interest (Hz)
        delays - 2-D array of MWA beamformer delay steps as numpy array shape (2,16),
                although a (16,) list or a (16,) array will also be accepted
                1st dimension is the antenna pol (NS, EW)
                2nd dimension is the antenna number
        amps - 2-D array of antenna amplitudes. These are absolute values
               (i.e. relatable to physical units)
        pixels_per_deg - angular resolution, being the number of pixels per degree along Az and ZA axes"""

        self.AA = AA

        if delays is None:
            delays = np.zeros([2, 16])

        if amps is None:
            amps = np.ones([2, 16])

        # Check valid amplitudes
        try:
            if isinstance(amps, list):
                amps = np.array(amps)
            if amps.shape == (16,):
                logger.warning('Assuming set of 16 antenna amplitudes are apply to both X Y dipoles')
                amps = np.tile(amps, (2, 1))
        except Exception:
            e = 'Unable to convert amplitudes "%s" to shape (2,16)' % (amps)
            logger.error(e)
            raise ValueError(e)
        if amps.shape != (2, 16):
            e = 'Amplitudes "%s" are not shape (2,16)' % (amps)
            logger.error(e)
            raise ValueError(e)

        #       #TODO: read number of antennas (i.e. delays) from h5 file
        # Check valid delays
        try:
            if isinstance(delays, list):
                delays = np.array(delays)
            if delays.shape == (16,):
                logger.warning('Assuming set of 16 antenna delays apply to both X and Y dipoles')
                delays = np.tile(delays, (2, 1))
        except Exception:
            e = 'Unable to convert delays "%s" to shape (2,16)' % (delays)
            logger.error(e)
            raise ValueError(e)
        if delays.shape != (2, 16):
            e = 'Delays "%s" are not shape (2,16)' % (delays)
            logger.error(e)
            raise ValueError(e)

        if (delays > 32).any():
            e = 'There are delays greater than 32: "%s"' % (delays)
            logger.error(e)
            raise ValueError(e)

        # check for terminated dipoles and reset delays and amps
        terminated = delays == 32
        if (terminated).any():
            logger.warning('Terminated dipoles (delay setting 32)... setting amplitude and delay to zero.')
            delays[terminated] = 0
            amps[terminated] = 0

        logger.info('Using delays X=%s, Y=%s' % (delays[0], delays[1]))
        self.delays = delays

        logger.info('Using amplitudes X=%s, Y=%s' % (amps[0], amps[1]))
        self.amps = amps
        self.beam_modes = {}
        self.calc_beam_modes()

    def calc_beam_modes(self):
        """Calculate (accumulate) modes for beam object initialised
        with delays and amplitudes"""
        self.beam_modes = {}
        pols = ['X', 'Y']
        for pol in [0, 1]:
            logger.debug('Calculate (accumulate) modes for %s-pol beam. Time is %s' % (pols[pol],
                                                                                       datetime.datetime.now().time()))
            # Calculate complex excitation voltages
            phases = 2 * math.pi * self.AA.freq * (-self.delays[pol]) * 435e-12  # convert delay to phase
            Vcplx = self.amps[pol] * np.exp(1.0j * phases)  # complex excitation col voltage

            # sum up modes to create beam described by spherical harmonics
            logger.debug('determine theta-dependent component...')

            # finding maximum length of modes for this frequency
            max_length = 0  # initialize
            n_ant = self.AA.n_ant  # Was hardcoded to 16
            for ant_i in range(n_ant):
                # select spherical wave table
                name = '%s%s_%s' % (pols[pol], ant_i + 1, self.AA.freq)

                # find maximum length
                if self.AA.h5f[name].shape[1] // 2 > max_length:
                    max_length = self.AA.h5f[name].shape[1] // 2

            # accumulating spherical harmonics coefficients for the array
            # initialize
            Q1_accum = np.zeros(max_length, dtype=np.complex128)
            Q2_accum = np.zeros(max_length, dtype=np.complex128)

            # Read in modes
            Q_modes_all = self.AA.h5f['modes'].value.T
            Nmax = 0
            M_accum = None
            N_accum = None
            for ant_i in range(n_ant):
                # re-initialise Q1 and Q2 for every antenna
                Q1 = np.zeros(max_length, dtype=np.complex128)
                Q2 = np.zeros(max_length, dtype=np.complex128)

                # select spherical wave table
                name = '%s%s_%s' % (pols[pol], ant_i + 1, self.AA.freq)
                Q_all = self.AA.h5f[name].value.T

                # current length
                my_len = np.max(Q_all.shape)
                my_len_half = my_len // 2

                Q_modes = Q_modes_all[0:my_len, :]  # Get modes for this antenna

                # convert Qall to M, N, Q1, Q2 vectors for processing

                # find s=1 and s=2 indices
                # only find s1 and s2 for this antenna
                s1 = np.array(Q_modes[0:my_len, 0] <= 1, dtype=int)
                s2 = np.array(Q_modes[0:my_len, 0] > 1, dtype=int)
                
                # grab m,n vectors
                M = Q_modes[s1, 1]
                N = Q_modes[s1, 2]

                # update to the larger M and N
                if np.max(N) > Nmax:
                    M_accum = M
                    N_accum = N
                    Nmax = np.max(N_accum)

                # grab Q1mn and Q2mn and make them complex
                for qi in range(my_len_half):
                    Q1[qi] = Q_all[s1[qi], 0] * np.exp(1.0j * Q_all[s1[qi], 1] * deg2rad)
                    Q2[qi] = Q_all[s2[qi], 0] * np.exp(1.0j * Q_all[s2[qi], 1] * deg2rad)

                # accumulate Q1 and Q2, scaled by excitation voltage
                Q1_accum += Q1 * Vcplx[ant_i]
                Q2_accum += Q2 * Vcplx[ant_i]
            self.beam_modes[pols[pol]] = {'Q1': Q1_accum, 'Q2': Q2_accum,
                                          'M': M_accum, 'N': N_accum}

    def get_response(self, phi_arr, theta_arr):
        """Calculate full Jones matrix response (E-field) of beam for
        one or more spherical coordinates

        Input:
        phi_arr and theta_arr are single values or arrays of equal shape

        phi_arr - azimuth angles (radians), north through east.
        theta_arr - zenith angles (radian)

        Output:
        Jones - A multi-dimensional array, comprising an array of shape(phi_arr),
        with [2][2] at the start for the Jones vectors, where
        [J_11=Xtheta J_12=Xphi]
        [J_21=Ytheta J_21=Yphi]
        """

        # ==============================================================================
        # Code showing that (Jones, az, za) is faster than (az, za, Jones)
        # a = np.random.rand(5000, 5000, 2, 2)
        # timeit -n 100 a[:,:,0,0].sum()
        # -> 100 loops, best of 3: 55.2 ms per loop
        # b = np.random.rand(2,2,5000, 5000)
        # timeit -n 100 b[0,0].sum()
        # -> 100 loops, best of 3: 18.1 ms per loop
        # ==============================================================================

        # Convert to numpy array (if not already numpy array)
        try:
            phi_arr = np.array(phi_arr, copy=False, ndmin=1)
            theta_arr = np.array(theta_arr, copy=False, ndmin=1)
        except Exception:
            e = 'Unable to convert theta and phi to numpy arrays'
            logger.error(e)
            raise ValueError(e)

        if phi_arr.ndim == 0 and theta_arr.ndim == 0:  # Convert single value to array
            phi_arr = np.reshape(phi_arr, (1))
            theta_arr = np.reshape(theta_arr, (1))
            # Calculate for each point
        logger.debug('Calculating beam for each point in %s... %s' % (phi_arr.shape, datetime.datetime.now().time()))
        Jones = self.get_FF(phi_arr, theta_arr, grid=False)

        return Jones

    def get_interp_response(self, phi_arr, theta_arr, pixels_per_deg=5):
        """Calculate full Jones matrix response (E-field) of beam interpolated
        from a beam calculated on a 2-D grid of spherical coordinates at
        resolution pixels_per_deg pixels per degree.
        Where the input is many unique theta and phi coordinates,
        this approach is faster than calculating the response for each
        unique coordinate.

        Input:
        phi_arr and theta_arr are arrays of equal shape defining points
        where the response is required

        phi_arr - azimuth angles (radians), north through east.
        theta_arr - zenith angles (radian)
        pixels_per_deg - number of pixels per degree along phi and theta axes
                        which is then interpolarted on the phi_arr,theta_arr coords

        Output:
        Jones - A 4-D array, comprising a 2-D array of shape(phi_arr),
        with [2][2] at the start for the Jones vectors, where
        [J_11=Xtheta J_12=Xphi]
        [J_21=Ytheta J_21=Yphi]
        """

        # RegularGridInterpolator only needed for this function.
        # FIXME: version check, as it's not available on earlier scipy versions

        # TODO: find min & max of phi_arr & theta_arr so that a subset of the
        # grid can be calculated (instead of the whole 0-360, 0-90 grid)

        # Convert to numpy array (if not already numpy array)
        try:
            phi_arr = np.array(phi_arr, copy=False, ndmin=1)
            theta_arr = np.array(theta_arr, copy=False, ndmin=1)
        except Exception:
            e = 'Unable to convert theta and phi to numpy arrays'
            logger.error(e)
            raise ValueError(e)

        # change negative azimuths to positive ones, otherwise it won't work with the interpolation function below
        phi_arr[phi_arr < 0] += 2.0 * math.pi
        
        # print "DEBUG = %s" % (phi_arr)

        logger.debug('Calculating a gridded beam and interpolating onto coordinates of shape %s...' % (phi_arr.shape,))
        # Interpolate from gridded beam
        Jones = np.zeros((2, 2) + np.shape(phi_arr), dtype=np.complex128)

        logger.debug('Calculating gridded beam (Az=0-360, ZA=0-90) at angular resolution %s pixels per degree... %s' %
                     (pixels_per_deg, datetime.datetime.now().time()))
        if pixels_per_deg < 5:
            logger.warning(
                "Resolution along theta, phi axes is less than 5 pixels per degree. Results may be less reliable")

            # Calculate beam for a phi (NtE), theta grid with angular resolution specified by pixels_per_deg.
        mygrid = get_grid('rad', pixels_per_deg)
        gridded_Jones = self.get_FF(mygrid['phi_1D'], mygrid['theta_1D'], grid=True)

        logger.debug('Interpolating... %s' % datetime.datetime.now().time())
        for i in [0, 1]:
            for ii in [0, 1]:
                # Interpolate real and imag separately (just to be sure) and reconstruct
                my_interp_fn_real = RegularGridInterpolator((mygrid['phi_1D'], mygrid['theta_1D']),
                                                            gridded_Jones[i, ii].real,
                                                            bounds_error=False)  # bounds_error=False interpolates NaNs to NaN
                my_real = my_interp_fn_real(np.dstack([phi_arr, theta_arr]))

                my_interp_fn_imag = RegularGridInterpolator((mygrid['phi_1D'], mygrid['theta_1D']),
                                                            gridded_Jones[i, ii].imag,
                                                            bounds_error=False)  # bounds_error=False interpolates NaNs to NaN
                my_imag = my_interp_fn_imag(np.dstack([phi_arr, theta_arr]))
                Jones[i, ii] = my_real + 1j * my_imag
        logger.debug('Done... %s' % datetime.datetime.now().time())

        return Jones

    def get_FF(self, phi_arr, theta_arr, grid):
        """
        Converts the beam object's spherical harmonics to a Jones matrix of
        an E-field (polarized in \hat{theta} and \hat{phi}).

        Input:
        phi_arr - Array of azimuth angles (radians), north through east
        theta_arr - Array of zenith angles
        grid - If True, will return a 2-D array based on input theta, phi.
               If False will return a array of size of input theta, phi.

        Output:
        #E_P - phi polarized field
        #E_T - theta polarazed field
        Sigma_T -  Sigma_T is theta polarized field without the
                   sqrt(Zo/(2pi))*7exp(-jbeta r)/r factor
        Sigma_P -  Similarly for Sigma_P"""

        if grid:
            # Create 4-D Jones matrix of shape: 2 x 2 x n_phi x n_theta
            if phi_arr.ndim != 1 and theta_arr.ndim != 1:
                e = 'For gridded beam, theta (shape %s) and phi (shape %s) must be 1-D arrays'
                logger.error(e % (np.shape(theta_arr), np.shape(phi_arr)))
                raise ValueError(e % (np.shape(theta_arr), np.shape(phi_arr)))
            Jones = np.zeros((2, 2, len(phi_arr), len(theta_arr)), dtype=np.complex128)
        else:
            # Create Jones matrix of shape: 2 x 2 x shape(phi_arr)
            if phi_arr.shape != theta_arr.shape:
                e = 'Theta (shape %s) and phi (shape %s) must be the same shape'
                logger.error(e % (np.shape(theta_arr), np.shape(phi_arr)))
                raise ValueError(e % (np.shape(theta_arr), np.shape(phi_arr)))
            Jones = np.zeros((2, 2) + np.shape(phi_arr), dtype=np.complex128)

        counter = 10000  # Counter for messages

        phi_arr = math.pi / 2 - phi_arr  # Convert to East through North (FEKO coords)
        phi_arr[phi_arr < 0] += 2 * math.pi  # 360 wrap

        pols = ['X', 'Y']
        for pol in [0, 1]:
            # Extract modes for this pol
            M = self.beam_modes[pols[pol]]['M']
            N = self.beam_modes[pols[pol]]['N']
            Q1 = self.beam_modes[pols[pol]]['Q1']
            Q2 = self.beam_modes[pols[pol]]['Q2']

            # form P(cos\theta)/(sin\theta) and P^{m+1}(cos\theta)with FEKO M,N order
            nmax = int(np.max(N))
            if np.max(N) - nmax != 0:
                logger.error('The maximum of N should be an integer value!')

            # form pre-multiplying constants in (1) of "Calculating...."
            # Rick 19-12-2016: this is fine for the amount of modes used for the MWA simulations
            # but the factorials explode for higher number of modes!
            C_MN = (0.5 * (2 * N + 1) * misc.factorial(N - abs(M)) / misc.factorial(N + abs(M))) ** 0.5
            # Rick 19-12-2016: remove annoying "RuntimeWarning: invalid value encountered in divide, MabsM=-M/np.abs(M) ", caused by:
            # MabsM=-M/np.abs(M)  <----
            # MabsM[MabsM==np.NaN]=1 #for M=0, replace NaN with MabsM=1;
            # MabsM=(MabsM)**M
            # this does the same trick: 1 for M<=0, -1 for odd M>0
            MabsM = np.ones(M.shape)
            MabsM[(M > 0) & (M % 2 != 0)] = -1

            if len(phi_arr.ravel()) > counter:
                logger.debug('Time is %s' % datetime.datetime.now().time())
                logger.warning('Calculating for %s points. This may take a while!' % len(phi_arr.ravel()))

            # determine unique thetas, phis to speed up calculations
            # also store the indices of the unique phi/theta components
            if not grid:
                phi_unique, phi_index = np.unique(phi_arr, return_inverse=True)
                theta_unique, theta_index = np.unique(theta_arr, return_inverse=True)  # speeds up calculations
            else:  # We expect all to be unique
                phi_unique, phi_index = phi_arr, None
                theta_unique, theta_index = theta_arr, None

            # Rick 16-3-2017
            # calculate phi-dependent component ( phi_comp ), but only for each unique M  (!!)
            # make sure data is stored as a contiguous array	to reduce cache misses
            # (should be the case automatically, but just to be sure)
            phi_comp = np.ascontiguousarray(np.exp(1.0j * np.outer(phi_unique, list(range(-nmax, nmax + 1)))))
            # determine theta-dependent components
            # nomenclature:
            # T and P are the sky polarisations theta and phi
            # theta and phi are direction coordinates

            (P_sin, P1) = P1sin_array(nmax, theta_unique)
            M_u = np.outer(np.cos(theta_unique), np.abs(M))
            phi_const = C_MN * MabsM / (N * (N + 1)) ** 0.5
            emn_T = (1.0j) ** N * (P_sin * (M_u * Q2 - M * Q1) + Q2 * P1) * phi_const
            emn_P = (1.0j) ** (N + 1) * (P_sin * (M * Q2 - Q1 * M_u) - Q1 * P1) * phi_const

            # Rick 19-12-2016: Use a matrix multiplication to calculate Emn_P and Emn_T.
            # Sum results of Emn_P and emn_T for each unique M
            # this will make the matrix multiplication much faster, especially for
            # really fine grids (dimension reduces from [nmax^2+2*nmax] to [2*nmax+1] )
            # tests showed that the overhead for summing and re-calculating phi
            # is smaller than doing the dot product without reductions (especially for fine grids)
            emn_P_sum = np.zeros((len(theta_unique), 2 * nmax + 1), dtype=np.complex128)
            emn_T_sum = np.zeros((len(theta_unique), 2 * nmax + 1), dtype=np.complex128)
            for m in range(-nmax, nmax + 1):
                emn_P_sum[:, m + nmax] = np.sum(emn_P[:, M == m], axis=1)
                emn_T_sum[:, m + nmax] = np.sum(emn_T[:, M == m], axis=1)

            if grid:  # Calculate via gridded approach
                # the actual calculation using dot product
                Sigma_P = np.inner(phi_comp, emn_P_sum)
                Sigma_T = np.inner(phi_comp, emn_T_sum)
                # print('calculate sigma done...%s'%datetime.datetime.now().time())
            else:  # Calculate for every value in theta,phi. could probably be vectorised too
                # determine whether it's more worth it to do a dot product for
                # each unique theta/phi combination (creates a 2D array of (len(theta_unique),len(phi_unique))
                # OR do a prodoct/sum of 2 arrays of (len(phi_arr) * (2*nmax+1))
                if (len(phi_unique) * len(phi_unique)) < (len(phi_arr.ravel()) * (nmax * 2 + 1)):
                    # calculate using an inner product and selecting the desired coordinates afterwards
                    # benificial for a low number of unique (theta,phi) coordinates, e.g. on a (semi) regular grid
                    Sigma_P_comp = np.inner(phi_comp, emn_P_sum)
                    Sigma_T_comp = np.inner(phi_comp, emn_T_sum)
                    Sigma_P = Sigma_P_comp[phi_index, theta_index].reshape(np.shape(phi_arr))
                    Sigma_T = Sigma_T_comp[phi_index, theta_index].reshape(np.shape(phi_arr))
                else:
                    # calculate by copying correct sections of data and doing a multiply/sum afterwards
                    # benificial in the case of many unique (theta,phi) coordinates, e.g. on a random grid
                    # construct 3 arrays of dimension (len(phi_arr),2*nmax+1)
                    emn_P_2 = emn_P_sum[theta_index, :]
                    emn_T_2 = emn_T_sum[theta_index, :]
                    phi_comp_2 = phi_comp[phi_index, :]
                    # print('preparing done, calculating sigma (ungridded, sum)...%s'%datetime.datetime.now().time())
                    # the calculation involves multiplying and summing each column
                    # the result is
                    Sigma_T = np.sum(emn_T_2 * phi_comp_2, axis=1).reshape(np.shape(phi_arr))
                    Sigma_P = np.sum(emn_P_2 * phi_comp_2, axis=1).reshape(np.shape(phi_arr))

            # to match with FEKO (neglects: exp(jbeta r)/r factor)
            logger.debug('Done... %s' % datetime.datetime.now().time())

            #    mu0=4*math.pi*1e-7
            #    eps0=8.85418781761e-12
            #    Zo=(mu0/eps0)**0.5
            #    sqrt_fac=(Zo/(2*math.pi))**0.5
            #
            #    E_P=sqrt_fac*Sigma_P
            #    E_T=sqrt_fac*Sigma_T

            # Save for this polarisation
            Jones[pol, 0] = Sigma_T
            Jones[pol, 1] = -Sigma_P  # 2017-05-30 : sign fixed by MS to reflect the fact that phi=90-az (it is not just change of values but orientation of base vector changes,
            # hence the sign of the Phi component of electric field has to change too. It was also fixed

        return Jones


# Rick 19-12-2016
# calculate P^|m|_n(cos(theta))/sin(theta) and P^(|m|+1)_n(cos(theta))
# similar to the "P1sin" function, but calculates for all theta in one go
# at the end of the function, patches are made using the original P1sin function
# to solve the 0/0 issue
def P1sin_array(nmax, theta):
    u = np.cos(theta)
    sin_theta = np.sin(theta)
    # Make sure that we don't divide by 0 (sin(0) = sin(pi) = 0 ) proper results
    # are inserted at the end of this function. Set to NaN for now
    sin_theta[(theta == 0) | (theta == math.pi)] = np.NaN
    # create at forehand
    P_sin = np.zeros((nmax ** 2 + 2 * nmax, np.size(theta)))
    P1 = np.zeros((nmax ** 2 + 2 * nmax, np.size(theta)))
    for n in range(1, nmax + 1):
        # legendre P_{n}^{|m|=0...n} (u)
        orders = np.arange(0, n + 1)
        orders = orders.reshape(n + 1, 1)
        # THESE ARE THE SAME:
        # legendre(2,0:0.1:0.2) (matlab)
        # scipy:
        # a=np.arange(0,3)
        # b=a.reshape(3,1)
        # lpmv(b,2,np.arange(0,0.3,0.1))
        # fetch entire matrix in one go (for a particular n)
        # in theory, fetching for all n in one go should also be possible
        P = lpmv(orders, n, u)
        # P_{n}^{|m|+1} (u)
        Pm1 = np.vstack([P[1::, :], np.zeros(
            (1, np.size(theta)))])  # I should just be able to use orders=np.arange(1,n+1), then append zero?
        # Pm1=Pm1.reshape(len(Pm1),1)   # FIXME: can probably make this and others 1-D
        # P_{n}^{|m|}(u)/sin_th
        # Pm_sin=np.zeros((n+1,1),dtype=np.complex128) #initialize
        # parameters
        Pm_sin = P / sin_theta
        # accumulate Psin and P1 for the m values
        ind_start = (n - 1) ** 2 + 2 * (n - 1)  # start index to populate
        ind_stop = n ** 2 + 2 * n  # stop index to populate
        # assign
        P_sin[np.arange(ind_start, ind_stop), :] = np.vstack([np.flipud(Pm_sin[1::, :]), Pm_sin])
        P1[np.arange(ind_start, ind_stop), :] = np.vstack([np.flipud(Pm1[1::, :]), Pm1])
        # fix for theta = 0 and theta = pi (properly handled in P1sin, so use that function )
    P_sin[:, theta == 0] = np.array([P1sin(nmax, 0)[0]]).transpose()
    P_sin[:, theta == math.pi] = np.array([P1sin(nmax, math.pi)[0]]).transpose()
    return (P_sin.transpose(), P1.transpose())


def P1sin(nmax, theta):
    """Create the Legendre function flavors for FF expansion using spherical wave
       See:
       Calculating Far-Field Radiation Based on FEKO Spherical Wave Coefficients,
       draft 10 June 2015
       14/07/2015: ATS - using slope estimator for u=1/-1 (forward/backward
       difference)

       Input:
       1. theta (rad) is the cos\theta or sin\theta arguments
       2. nmax is maximum n from FEKO Q1mn and Q2mn, n must be >=1

       Output:
       1. P_sin: P_{n}^{|m|}(cos\theta)/sin(theta) with FEKO order M,N
       1. P1: P_{n}^{|m|+1}(cos\theta) with FEKO order M,N
    """

    # initialize for nmax, we have 2(1+...+nmax)+nmax=nmax^2+2*nmax long array
    # Rick 16-3-2017 complex128 is a bit overkill here
    P_sin = np.zeros((nmax ** 2 + 2 * nmax))
    P1 = P_sin * 0  # copy

    # theta arguments
    u = np.cos(theta)
    sin_th = np.sin(theta)
    delu = 1e-6  # for slope estimation

    # step from 1 to nmax
    for n in range(1, nmax + 1):
        # legendre P_{n}^{|m|=0...n} (u)
        orders = np.arange(0, n + 1)
        orders = orders.reshape(n + 1, 1)
        P = lpmv(orders, n, u)

        # THESE ARE THE SAME:
        # legendre(2,0:0.1:0.2) (matlab)
        # scipy:
        # a=np.arange(0,3)
        # a=a.reshape(3,1)
        # lpmv(b,2,np.arange(0,0.3,0.1))

        # P_{n}^{|m|+1} (u)
        Pm1 = np.append(P[1::], 0)  # I should just be able to use orders=np.arange(1,n+1), then append zero?
        Pm1 = Pm1.reshape(len(Pm1), 1)  # FIXME: can probably make this and others 1-D
        # P_{n}^{|m|}(u)/sin_th
        # Rick 20-12-2016 complex128 is not really needed here..
        Pm_sin = np.zeros((n + 1, 1))  # initialize
        # parameters
        # l = np.arange(0, n / 2 + 1)

        if u == 1:
            # special treatment depending on m;
            # for m=0, m=0 Pm_sin=inf so, the product m*Pm_sin is zero;
            # for m=1, we need a substitution
            # approach 1: based on E-9 in Harrington, this is not stable
            # for n>~45
            # Pm_sin(2,1)=-sum(((-1).^l.*factorial(2.*n-2.*l).*(n-2.*l))...
            #    ./(2.^n.*factorial(l).*factorial(n-l).*factorial(n-2.*l)));
            # approach 2: based on slope estimate
            # Pn(cos x)/sin x = -dPn(u)/du
            Pu_mdelu = lpmv(orders, n, u - delu)
            Pm_sin[1, 0] = -(P[0] - Pu_mdelu[0]) / delu  # backward difference

            # m>=2, value is 0, so initial values are OK
        elif u == -1:
            # approach 1: based on E-9 in Harrington, this is not stable
            # for n>~45
            # Pm_sin(2,1)=-sum(((-1).^l.*factorial(2.*n-2.*l).*(n-2.*l).*(-1).^(n-2.*l-1))...
            #    ./(2.^n.*factorial(l).*factorial(n-l).*factorial(n-2.*l)));
            # approach 2: based on slope estimate
            # Pn(cos x)/sin x = -dPn(u)/du
            Pu_mdelu = lpmv(orders, n, u - delu)
            Pm_sin[1, 0] = -(Pu_mdelu[0] - P[0]) / delu  # forward difference
        else:
            Pm_sin = P / sin_th

        # accumulate Psin and P1 for the m values
        ind_start = (n - 1) ** 2 + 2 * (n - 1)  # start index to populate
        ind_stop = n ** 2 + 2 * n  # stop index to populate
        # assign
        P_sin[np.arange(ind_start, ind_stop)] = np.append(np.flipud(Pm_sin[1::, 0]), Pm_sin)
        P1[np.arange(ind_start, ind_stop)] = np.append(np.flipud(Pm1[1::, 0]), Pm1)
    return (P_sin, P1)


def get_grid(unit, pixels_per_deg):
    """Return phi (Az), theta (ZA) grid
    Input:
    unit - 'deg' or 'rad'
    pixels_per_deg - number of pixels per degree along phi and theta axes"""

    logger.debug('Setting up phi (Az), theta (ZA) grid %s pixels per deg' % pixels_per_deg)
    degs_per_pixel = 1. / pixels_per_deg
    n_phi = int(360 / degs_per_pixel) + 1
    n_theta = int(90 / degs_per_pixel) + 1
    logger.debug('%s pixels on phi axis' % n_phi)
    logger.debug('%s pixels on theta axis' % n_theta)

    theta_1D = np.arange(0, n_theta) * degs_per_pixel
    phi_1D = np.arange(0, n_phi) * degs_per_pixel
    if unit == 'rad':
        theta_1D *= deg2rad
        phi_1D *= deg2rad
    theta = np.tile(theta_1D, (n_phi, 1))
    phi = (np.tile(phi_1D, (n_theta, 1))).T

    return {'theta': theta, 'phi': phi, 'theta_1D': theta_1D, 'phi_1D': phi_1D}


if __name__ == "__main__":

    logger.setLevel(logging.DEBUG)
    h5filepath = config.h5file  # recent version was MWA_embedded_element_pattern_V02.h5
    target_freq_Hz = 150e6
    logger.debug('Initialising ApertureArray object with h5filepath = %s' % h5filepath)
    tile = ApertureArray(h5filepath, target_freq_Hz)
    #    tile.calc_zenith_norm_fac()

    #    my_Astro_Az=26.5651
    #    my_ZA=15.3729
    #    delays=np.array([6, 7, 8, 9, 4, 5, 6, 7, 2, 3, 4, 5, 0, 1, 2, 3])
    ##
    #    my_Astro_Az=0
    #    my_ZA=28
    #    delays=np.array([6,6,6,6,4,4,4,4,2,2,2,2,0,0,0,0])
    #
    my_Astro_Az = 0
    my_ZA = 0
    delays = np.zeros([2, 16])  # Dual-pol.

    amps = np.ones([2, 16])

    logger.debug('Set up beam object')
    mybeam = Beam(tile, delays, amps=amps)

    logger.debug('Get Az,El points')
    pixels_per_deg = 2
    mygrid = get_grid('rad', pixels_per_deg)
    az = mygrid['phi']
    za = mygrid['theta']
    logger.debug('Interpolate from beam')
    Jones = mybeam.get_interp_response(az, za)
    logger.debug('Normalise to zenith')
    Jones = tile.apply_zenith_norm_Jones(Jones)

    logger.debug('Plot and save')
    my_Astro_Az = '%.0f' % my_Astro_Az
    my_ZA = '%.0f' % my_ZA

    point_dirn = 'Az%s-ZA%s' % (my_Astro_Az, my_ZA)
    filebase = '%s-%sPixPerDeg-%s' % (point_dirn, pixels_per_deg, True)
    title = 'Az_NtE=%s, ZA=%s\n' % (my_Astro_Az, my_ZA)

    # Get cut at pointing direction
    idx = az == (float(my_Astro_Az) * deg2rad)
    cut_1D = Jones[:, :, idx]

    # Plot jones matrices
    beam_tools.plotArrayJones(Jones, target_freq_Hz, filebase, title, pixels_per_deg, cut_1D)

    # Plot power for XX and YY
    beam_tools.plotVisResponse(Jones, target_freq_Hz, filebase, title, pixels_per_deg, gridded=True)

    # Export to .mat file for verification against matlab code
    beam_tools.exportArrayJones(Jones, target_freq_Hz, filebase)

    # Project onto hemishpere and re-run plots and write to fits files
    logger.debug('Project beam on hemisphre')
    proj = 'SIN'
    [az, za] = beam_tools.makeAZZA(1000, proj)
    grid = True
    pixels_per_deg = 5
    Jones = mybeam.get_interp_response(az, za, pixels_per_deg=pixels_per_deg)
    Jones = tile.apply_zenith_norm_Jones(Jones)  # Normalise

    writetofits = False
    if writetofits:
        vis = beam_tools.makeUnpolInstrumentalResponse(Jones, Jones)

        filename = 'beam_%sMHz_%s_E-W.fits' % (target_freq_Hz / 1e6, proj)
        fits.writeto(filename, np.abs(vis[0, 0]))
        filename = 'beam_%sMHz_%s_N-S.fits' % (target_freq_Hz / 1e6, proj)
        fits.writeto(filename, np.abs(vis[1, 1]))
        filename = '%s_az.fits' % (proj)
        fits.writeto(filename, az)
        filename = '%s_ZA.fits' % (proj)
        fits.writeto(filename, za)

    filebase = '%s-%sPixPerDeg-%s' % (point_dirn, pixels_per_deg, grid)
    beam_tools.plotArrayJones(Jones, target_freq_Hz, filebase + '_' + proj, title, pixels_per_deg)
    beam_tools.plotVisResponse(Jones, target_freq_Hz, filebase + '_' + proj, title, pixels_per_deg)
