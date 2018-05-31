#!/usr/bin/env python
"""Tools for calculating the complex voltage response of MWA tiles and dipoles
Randall Wayth. March 2014, based on the work of Adrian Sutinjo.
"""

import logging

import numpy

import astropy.io.fits as pyfits

from scipy import interpolate

import config
import mwa_impedance

vel_light = 2.99792e8
DQ = 435e-12 * vel_light  # delay quantum in distance light travels for 1 quantum
logging.basicConfig(format='# %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)  # default logger level is WARNING

# locations of dipoles in tile
dipole_sep = config.DIPOLE_SEPARATION  # meters
x_offsets = numpy.array([-1.5, -0.5, 0.5, 1.5,
                         -1.5, -0.5, 0.5, 1.5,
                         -1.5, -0.5, 0.5, 1.5,
                         -1.5, -0.5, 0.5, 1.5],
                        dtype=numpy.float32) * dipole_sep
y_offsets = numpy.array([1.5, 1.5, 1.5, 1.5,
                         0.5, 0.5, 0.5, 0.5,
                         -0.5, -0.5, -0.5, -0.5,
                         -1.5, -1.5, -1.5, -1.5],
                        dtype=numpy.float32) * dipole_sep


class Dipole(object):
    """Provides a generic dual pol dipole object to support MWA beam models"""

    def __init__(self,
                 atype='short',
                 height=0.29,
                 length=0.74,
                 lookup_filename=config.Jmatrix,
                 gain=numpy.eye(2, dtype=numpy.complex64)):
        """
          General dipole object. Dual pol crossed dipole.
          Assumes a groundscreen with dipole height meters above ground.
          Supported types are 'short': analytic short dipole
                              'long':  analytic real length dipole
                              'lookup': lookup table
          For a short dipole, the length of the dipole is ignored.
          Gain is a 2x2 matrix with gain and direction independent crosstalk
          for the X (upper) and Y (lower) dipole resectively
        """

        assert (atype == 'short' or atype == 'long' or atype == 'lookup'), 'Unknown atype %r' % atype
        self.atype = atype
        self.height = height
        self.length = length
        self.gain = gain
        self.interp_freq = 0.0
        self.lookup = None
        self.lookup_za = None
        self.lookup_ph = None
        self.i00_real, self.i01_real, self.i10_real, self.i11_real = (None, None, None, None)
        self.i00_imag, self.i01_imag, self.i10_imag, self.i11_imag = (None, None, None, None)
        self.j00norm, self.j01norm, self.j10norm, self.j11norm = (None, None, None, None)
        self.freqs = numpy.array([])
        if atype == 'lookup':
            self.loadLookup(lookup_filename)

    def loadLookup(self, lookup_filename):
        """Load a dipole Jones response lookup table (FITS file)"""
        # data is a direct conversion of the output of the simulation so has redundant info.
        # we do all the conversion and stuff here
        # for reference, columns are:
        # theta phi  real(Jxt(t,p)) imag(Jxt(t,p)) real(Jxp(t,p)) imag(Jxp(t,p)) real(Jyt(t,p)) imag(Jyt(t,p)) real(Jyp(t,p)) imag(Jyp(t,p)))
        try:
            logger.debug('Loading %s' % lookup_filename)
            hdulist = pyfits.open(lookup_filename)
        except IOError:
            raise Exception('Cannot load Jones matrix file %s' % lookup_filename)
        nfreqs = len(hdulist)
        self.lookup_za = numpy.unique(hdulist[0].data[:, 0])  # zenith angle
        self.lookup_ph = numpy.unique(hdulist[0].data[:, 1])  # phi angle == 90 - az
        # p = numpy.where
        nza = len(self.lookup_za)
        nph = len(self.lookup_ph)
        self.lookup = numpy.empty((nfreqs, nza, nph, 2, 2), dtype=numpy.complex64)
        freqs = []
        for i in range(nfreqs):
            hdu = hdulist[i]
            logger.debug('Loading J lookup matrix for freq ' + str(hdu.header['FREQ']))
            freqs.append(hdu.header['FREQ'])
            jxt = hdu.data[:, 2] + 1.0j * hdu.data[:, 3]
            jxp = hdu.data[:, 4] + 1.0j * hdu.data[:, 5]
            jyt = hdu.data[:, 6] + 1.0j * hdu.data[:, 7]
            jyp = hdu.data[:, 8] + 1.0j * hdu.data[:, 9]
            self.lookup[i, :, :, 0, 0] = jxt.reshape((nph, nza)).transpose()
            self.lookup[i, :, :, 0, 1] = jxp.reshape((nph, nza)).transpose()
            self.lookup[i, :, :, 1, 0] = jyt.reshape((nph, nza)).transpose()
            self.lookup[i, :, :, 1, 1] = jyp.reshape((nph, nza)).transpose()
        logger.debug('Loaded dipole Jones matrix lookup model from ' + lookup_filename + ' with ' + str(nfreqs) + ' freqs')
        self.freqs = numpy.array(freqs)
        logger.debug('Supported frequencies (MHz): ' + str(self.freqs / 1e6))
        logger.debug("There are " + str(nza) + " tabulated zenith angles: " + str(self.lookup_za))
        logger.debug("There are " + str(nph) + " tabulated phi angles: " + str(self.lookup_ph))

    def getJones(self, az, za, freq, zenith_norm=True):
        """Return the Jones matrix for arrays of az/za for a given freq
        az and za are numpy arrays of equal length.
        Results are in corrds of az/za unit vectors such that
        output[j,i,:,:] is a 2x2 Jones matrix that maps
        za onto E-W     az onto E-W
        za onto N-S     az onto N-S
        By default, normalise to the zenith
        """
        if self.atype == 'short':
            return self.getJonesShortDipole(az, za, freq, zenith_norm=zenith_norm)
        elif self.atype == 'lookup':
            return self.getJonesLookup(az, za, freq)
        else:
            raise Exception("Dipole atype %r is not implemented yet." % self.atype)

    def getJonesLookup(self, az, za, freq):
        """Return the Jones matrix for arrays of az/za for a given freq (Hz)
        this method interpolates from the tablulated numerical results loaded
        by the constructor"""
        # need to interpolate each of the 4 Jones elements separately and each
        # the real and imag separately (since interpolate.RectBivariateSpline)
        # apparently doesn't handle complex

        # find the nearest freq lookup table
        pos = numpy.argmin(numpy.abs(self.freqs - freq))
        if numpy.abs(self.freqs[pos] - freq) > 2e6:
            logger.warning("Nearest tabulated impedance matrix freq is more than 2 MHz away from desired freq.")
        logger.info("Selecting matrix for nearest freq " + str(self.freqs[pos]))

        # cache the interpolation functions
        if self.interp_freq != freq:
            self.interp_freq = freq
            logger.debug("Setting new cache lookup freq to " + str(self.freqs[pos]))
            self.i00_real = interpolate.RectBivariateSpline(self.lookup_za, self.lookup_ph,
                                                            self.lookup[pos, :, :, 0, 0].real)
            self.i00_imag = interpolate.RectBivariateSpline(self.lookup_za, self.lookup_ph,
                                                            self.lookup[pos, :, :, 0, 0].imag)
            self.i01_real = interpolate.RectBivariateSpline(self.lookup_za, self.lookup_ph,
                                                            self.lookup[pos, :, :, 0, 1].real)
            self.i01_imag = interpolate.RectBivariateSpline(self.lookup_za, self.lookup_ph,
                                                            self.lookup[pos, :, :, 0, 1].imag)
            self.i10_real = interpolate.RectBivariateSpline(self.lookup_za, self.lookup_ph,
                                                            self.lookup[pos, :, :, 1, 0].real)
            self.i10_imag = interpolate.RectBivariateSpline(self.lookup_za, self.lookup_ph,
                                                            self.lookup[pos, :, :, 1, 0].imag)
            self.i11_real = interpolate.RectBivariateSpline(self.lookup_za, self.lookup_ph,
                                                            self.lookup[pos, :, :, 1, 1].real)
            self.i11_imag = interpolate.RectBivariateSpline(self.lookup_za, self.lookup_ph,
                                                            self.lookup[pos, :, :, 1, 1].imag)
            # determine normalisation factors. The simulations include all ph angles at za=0.
            # these are not redundant, and the ph value determines the unit vector directions of
            # both axes. We should normalise by where the result will be maximal.
            # For the E-W dipoles, the projection of the ZA unit vec will be max when
            # pointing east, i.e. when ph=0. For the PH unit vec, this will be when ph=-90 or 90
            # For the N-S dipoles, projection of ZA onto N-S is max az ph=90 and
            # proj of ph onto N-S is max when ph=0

            # determine the indices of 0 and 90 degrees of phi in the tabulated values
            ph0 = numpy.where(self.lookup_ph == 0.0)
            ph90 = numpy.where(self.lookup_ph == 90.0)
            th0 = numpy.where(self.lookup_za == 0.0)

            self.j00norm = self.lookup[pos, th0, ph0, 0, 0]
            self.j01norm = -self.lookup[pos, th0, ph90, 0, 1]  # use -90, not 90.
            self.j10norm = self.lookup[pos, th0, ph90, 1, 0]
            self.j11norm = self.lookup[pos, th0, ph0, 1, 1]

        ph_deg = 90.0 - az * 180.0 / numpy.pi  # ph in degrees
        za_deg = za * 180.0 / numpy.pi
        p = ph_deg < 0
        ph_deg[p] += 360.0
        j00 = self.i00_real.ev(za_deg.flatten(), ph_deg.flatten()) + 1.0j * self.i00_imag.ev(za_deg.flatten(),
                                                                                             ph_deg.flatten())
        j01 = self.i01_real.ev(za_deg.flatten(), ph_deg.flatten()) + 1.0j * self.i01_imag.ev(za_deg.flatten(),
                                                                                             ph_deg.flatten())
        j10 = self.i10_real.ev(za_deg.flatten(), ph_deg.flatten()) + 1.0j * self.i10_imag.ev(za_deg.flatten(),
                                                                                             ph_deg.flatten())
        j11 = self.i11_real.ev(za_deg.flatten(), ph_deg.flatten()) + 1.0j * self.i11_imag.ev(za_deg.flatten(),
                                                                                             ph_deg.flatten())

        result = numpy.empty((za.shape + (2, 2)), dtype=numpy.complex64)
        result[..., 0, 0] = j00.reshape(za.shape) / self.j00norm
        result[..., 0, 1] = -j01.reshape(za.shape) / self.j01norm  # sign flip between az and phi
        result[..., 1, 0] = j10.reshape(za.shape) / self.j10norm
        result[..., 1, 1] = -j11.reshape(za.shape) / self.j11norm  # sign flip between az and phi
        return result

    def getJonesShortDipole(self, az, za, freq, zenith_norm=True):
        """Calculate the Jones matrix for a short dipole.
        This is defined by purely geometric projection of unit vectors
        on the sky onto the unit vector defined by the dipole's direction.
        """
        # check for input scalars vs arrays. If the input is scalar, then
        # convert it so that numpy array-based operations are possible
        if numpy.isscalar(az):
            logger.debug("Converting scalar az input to a matrix")
            az = numpy.asarray(az, dtype=numpy.float32)
        if numpy.isscalar(za):
            logger.debug("Converting scalar za input to a matrix")
            za = numpy.asarray(za, dtype=numpy.float32)
        assert az.shape == za.shape, "Input za and az arrays must have same dimension"

        # output array has 2x2 Jones matrix for every az/za point in input
        result = numpy.empty((za.shape + (2, 2)), dtype=numpy.complex64)
        # apply the groundscreen factor, which is independent of az
        # znorm = 1.0
        gs = self.groundScreen(za, freq)
        if zenith_norm:
            logger.debug("Normalising response to zenith")
            gs /= self.groundScreen(0.0, freq)
        result[..., 0, 0] = numpy.cos(za) * numpy.sin(az) * gs
        result[..., 0, 1] = numpy.cos(az) * gs
        result[..., 1, 0] = numpy.cos(za) * numpy.cos(az) * gs
        result[..., 1, 1] = -numpy.sin(az) * gs
        return result

    def groundScreen(self, za, freq):
        """
        Calculate the groundscreen effect for an ideal infinite groundscreen
        given the dipole's height above the screen and the frequency (Hz)
        """
        ll = vel_light / freq
        return numpy.sin(numpy.pi * (2.0 * self.height / ll) * numpy.cos(za)) * 2.0

    def __str__(self):
        return "Dipole. Type: " + self.atype + ". height: " + str(self.height) + "m. Gain: " + str(self.gain)


class ApertureArray(object):
    """Aperture array antenna object"""

    def __init__(self, dipoles=None, xpos=x_offsets, ypos=y_offsets):
        """Constructor for aperture array station. xpos and ypos are arrays with
        the coords of the dipoles (meters) in local coords relative to centre of
        the antenna looking down on the station. Ordering goes left to right, top to bottom,
        hence are offsets in east and north from the array phase centre.
        """
        assert dipoles is None or len(dipoles) == 16, "Expecting 16 input dipoles, got %r" % str(len(dipoles))
        if dipoles is None:
            d = Dipole(atype='short')
            dipoles = [d] * 16
        self.dipoles = dipoles
        self.xpos = xpos
        self.ypos = ypos
        self.im = mwa_impedance.TileImpedanceMatrix()
        self.lna_z = mwa_impedance.LNAImpedance()

    def getPortCurrents(self, freq, delays=numpy.zeros((2, 16), dtype=numpy.float32)):
        """
        Return the port currents on a tile given the freq (Hz) and delays (integer)
        """
        lam = vel_light / freq
        phases = -2.0 * numpy.pi * delays * (DQ / lam)
        ph_rot = numpy.cos(phases) + 1.0j * numpy.sin(phases)
        # this code ignores any dipole gain (and crosstalk) terms.
        # should FIXME it.
        z_total = self.im.getImpedanceMatrix(freq) + numpy.eye(32) * self.lna_z.getZ(freq)
        inv_z = numpy.linalg.inv(z_total)
        port_current = numpy.dot(inv_z, ph_rot.reshape(32)).reshape(2, 16)
        return port_current

    def getArrayFactor(self, az, za, freq=155e6, delays=numpy.zeros((2, 16), dtype=numpy.float32)):
        """
        Get the scalar array factor response of the array for a given
        freq (Hz) and delay settings.
        az and za (radian) are numpy arrays of equal length defining a set
        of points to calculate the response for.
        delays is a 2D array of integer delay steps for the Y and X pol
        respectively.
        Result are in same coords as the az/za input arrays
        """
        lam = vel_light / freq
        port_current = self.getPortCurrents(freq, delays)

        # check for input scalars vs arrays. If the input is scalar, then
        # convert it so that numpy array-based operations are possible
        if numpy.isscalar(az):
            logger.debug("Converting scalar az input to a matrix")
            az = numpy.matrix(az, dtype=numpy.float32)
        if numpy.isscalar(za):
            logger.debug("Converting scalar za input to a matrix")
            za = numpy.matrix(za, dtype=numpy.float32)
        assert az.shape == za.shape, "Input az and za arrays must have same dimenions"

        # now calculate the array factor using these port currents
        sz = numpy.sin(za)
        kx = (2.0 * numpy.pi / lam) * numpy.sin(az) * sz
        ky = (2.0 * numpy.pi / lam) * numpy.cos(az) * sz
        ax = numpy.zeros_like(az, dtype=numpy.complex64)
        ay = numpy.zeros_like(az, dtype=numpy.complex64)
        for i in range(len(self.xpos)):
            ph = kx * self.xpos[i] + ky * self.ypos[i]
            ax += port_current[1, i] * (numpy.cos(ph) + 1.0j * numpy.sin(ph))  # X dipoles
            ay += port_current[0, i] * (numpy.cos(ph) + 1.0j * numpy.sin(ph))  # Y dipoles
        # set the points below the horizon to zero
        p = za >= numpy.pi / 2.0
        ax[p] = 0.0
        ay[p] = 0.0
        return (ax, ay)

    def getResponse(self, az, za, freq=155e6, delays=None):
        """
        Get the full Jones matrix response of the tile including the dipole
        reponse and array factor incorporating any mutual coupling effects
        from the impedance matrix. freq in Hz.
        delays in unit steps of beamformer delays as numpy array shape (2,16).
        az and za (radian) are numpy arrays of equal length defining a set
        of points to calculate the response for.
        Result is an array like az/za with [2][2] on the end for the Jones.
        """
        assert delays is None or numpy.size(delays) == 32, "Expecting 32 delays, got %r" % str(numpy.size(delays))
        if delays is None:
            delays = numpy.zeros((2, 16), dtype=numpy.float32)
        (ax, ay) = self.getArrayFactor(az, za, freq, delays)
        # get the zenith response to normalise to:
        (zax, zay) = self.getArrayFactor(numpy.array([0.0]), numpy.array([0.0]), freq)  # no delays == zenith
        ax /= numpy.abs(zax)
        ay /= numpy.abs(zay)
        d = self.dipoles[0]  # for now, assume all dipoles identical FIXME
        j = d.getJones(az, za, freq)
        j[..., 0, 0] *= ax
        j[..., 0, 1] *= ax
        j[..., 1, 0] *= ay
        j[..., 1, 1] *= ay
        return j


def convertJonesAzEl2HaDec(az, za, dec, lat):
    """
    Generate a converter for arrays of Jones matrices in Az/ZA to HA/DEC.
    This is a rotation of the orthogonal az/za unit vectors to
    the orthogonal ha/dec unit vectors by the parallactic angle
    plus some negative signs to account for the definition of
    coord axes. Multiply this by your az/za Jones to get a HA/DEC Jones.
    Inputs and outputs are numpy arrays of dimension [az][za][2][2]"""
    # parallactic angle needs HA, DEC and lat.
    (ha, dec) = h2e(az, za, lat)
    pa = calcParallacticAngle(ha, dec, lat)
    rot = numpy.empty((ha.shape + (2, 2)))
    crot = numpy.cos(pa * numpy.pi / 180.0)
    srot = numpy.sin(pa * numpy.pi / 180.0)
    # for clockwise rotations, use [[cos,sin],[-sin,cos]]
    rot[:, :, 0, 0] = crot
    rot[:, :, 0, 1] = srot
    rot[:, :, 1, 0] = -srot
    rot[:, :, 1, 1] = crot


def h2e(az, za, lat):
    """
    Horizon to equatorial.
    Convert az/za (radian) to HA/DEC (degrees, degrees)
    given an observatory latitude (degrees)
    """
    sa = numpy.sin(az)
    ca = numpy.cos(az)
    se = numpy.sin(numpy.pi / 2.0 - za)
    ce = numpy.cos(numpy.pi / 2.0 - za)
    sp = numpy.sin(lat * numpy.pi / 180.0)
    cp = numpy.cos(lat * numpy.pi / 180.0)

    # HA,Dec as x,y,z */
    x = - ca * ce * sp + se * cp
    y = - sa * ce
    z = ca * ce * cp + se * sp

    # To spherical */
    r = numpy.sqrt(x * x + y * y)
    ha = numpy.arctan2(y, x) * 180.0 / numpy.pi
    dec = numpy.arctan2(z, r) * 180.0 / numpy.pi
    return (ha, dec)


def e2h(ha, dec, lat):
    """
    Equatorial to horizon.
    Convert equatorial ha/dec coords (both in degs) to az,za (radian)
    given an observer latitute (degs). Returns (az,za)
    """
    ha_rad = ha * numpy.pi / 180.0
    dec_rad = dec * numpy.pi / 180.0
    lat_rad = lat * numpy.pi / 180.0
    sh = numpy.sin(ha_rad)
    ch = numpy.cos(ha_rad)
    sd = numpy.sin(dec_rad)
    cd = numpy.cos(dec_rad)
    sp = numpy.sin(lat_rad)
    cp = numpy.cos(lat_rad)
    x = - ch * cd * sp + sd * cp
    y = - sh * cd
    z = ch * cd * cp + sd * sp
    r = numpy.sqrt(x * x + y * y)
    a = numpy.arctan2(y, x)
    if a < 0.0:
        az = a + 2.0 * numpy.pi
    else:
        az = a
    el = numpy.arctan2(z, r)
    return (az, numpy.pi / 2.0 - el)


def calcParallacticAngle(ha, dec, lat):
    """
    Calculate the parallactic angle in degrees given an HA (degs)
    dec (degrees) and observatory latitude (degrees)
    """
    cl = numpy.cos(lat * numpy.pi / 180.0)
    sl = numpy.sin(lat * numpy.pi / 180.0)
    ch = numpy.cos(ha * numpy.pi / 180.0)
    sh = numpy.sin(ha * numpy.pi / 180.0)
    cd = numpy.cos(dec * numpy.pi / 180.0)
    sd = numpy.sin(dec * numpy.pi / 180.0)
    num = cl * sh
    den = sl * cd - cl * sd * ch
    return numpy.arctan2(num, den) * 180.0 / numpy.pi


def makeAZZA(npix=256):
    """
    Make azimuth and zenith angle arrays for a square image of side npix
    Projection is sine, all-sky
    Returns (az,za). Angles are in radian.
    """
    # build az and za arrays
    z = numpy.arange(npix, dtype=numpy.float32) - npix / 2
    x = numpy.empty((npix, npix), dtype=numpy.float32)
    y = numpy.empty((npix, npix), dtype=numpy.float32)
    for i in range(npix):
        y[i, 0:] = z
        x[0:, i] = z
    d = numpy.sqrt(x * x + y * y) / (npix / 2)
    # only select pixels above horizon
    t = (d <= 1.0)
    za = numpy.ones((npix, npix), dtype=numpy.float32) * numpy.pi / 2.0
    za[t] = numpy.arcsin(d[t])
    az = numpy.arctan2(y, x)
    return az, za


def makeUnpolInstrumentalResponse(j1, j2):
    """
    Form the visibility matrix in instrumental response from two Jones
    matrices assuming unpolarised sources (hence the brightness matrix is
    the identity matrix)
    Input: j1,j2: Jones matrices of dimension[za][az][2][2]
    Returns: [za][az][[xx,xy],[yx,yy]] where "X" and "Y" are defined by the receptors
    of the Dipole object used in the ApertureArray. Hence to get "XX", you want
    result[za][az][0][0] and for "YY" you want result[za][az][1][1]
    """
    result = numpy.empty_like(j1)

    result[:, :, 0, 0] = j1[:, :, 0, 0] * j2[:, :, 0, 0].conjugate() + j1[:, :, 0, 1] * j2[:, :, 0, 1].conjugate()
    result[:, :, 1, 1] = j1[:, :, 1, 0] * j2[:, :, 1, 0].conjugate() + j1[:, :, 1, 1] * j2[:, :, 1, 1].conjugate()
    result[:, :, 0, 1] = j1[:, :, 0, 0] * j2[:, :, 1, 0].conjugate() + j1[:, :, 0, 1] * j2[:, :, 1, 1].conjugate()
    result[:, :, 1, 0] = j1[:, :, 1, 0] * j2[:, :, 0, 0].conjugate() + j1[:, :, 1, 1] * j2[:, :, 0, 1].conjugate()
    return result


# def makePolInstrumentResponse(j1, j2, b):
#   """
#   Form the instrument response from two Jones matrices with an
#   arbitrary source brightness matrix, hence arbitrary polarisation
#   Returns: (xx,yy,xy,yx) where "X" and "Y" are defined by the receptors
#   of the Dipole object used in the ApertureArray
#   """
#   # FIXME: need to work out how to do this in vectorised way.


def plotDipoleJones(d, freq=155e6):
    """
    Utility to make a plot of a dipole Jones matrix for debugging
    """

    import matplotlib.pyplot as plt
    (az, za) = makeAZZA()
    logger.info("plotting dipole Jones response for atype: " + d.atype + ', freq (MHz): ' + str(freq / 1e6))
    j = d.getJones(az, za, freq)

    plt.imshow(j[:, :, 0, 0].real)
    plt.title('MWA ' + str(freq / 1e6) + 'MHz dipole J00 voltage real')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_J00_voltage_re_' + str(freq / 1e6) + 'MHz_dipole.png')
    plt.clf()

    plt.imshow(j[:, :, 0, 1].real)
    plt.title('MWA ' + str(freq / 1e6) + 'MHz dipole J01 voltage real')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_J01_voltage_re_' + str(freq / 1e6) + 'MHz_dipole.png')
    plt.clf()

    plt.imshow(j[:, :, 1, 0].real)
    plt.title('MWA ' + str(freq / 1e6) + 'MHz dipole J10 voltage real')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_J10_voltage_re_' + str(freq / 1e6) + 'MHz_dipole.png')
    plt.clf()

    plt.imshow(j[:, :, 1, 1].real)
    plt.title('MWA ' + str(freq / 1e6) + 'MHz dipole J11 voltage real')
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_J11_voltage_re_' + str(freq / 1e6) + 'MHz_dipole.png')
    plt.clf()


def plotArrayJones(j, freq, za):
    """
    Utility to plot the output of tile Jones matrices
    """
    import matplotlib.pyplot as plt

    plt.imshow(numpy.abs(j[:, :, 0, 0]))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz J00 voltage mag ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_J00_voltage_mag_' + str(freq / 1e6) + 'MHz_ZA' + str(za) + '.png')
    plt.clf()

    plt.imshow(numpy.abs(j[:, :, 0, 1]))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz J01 voltage mag ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_J01_voltage_mag_' + str(freq / 1e6) + 'MHz_ZA' + str(za) + '.png')
    plt.clf()

    plt.imshow(numpy.abs(j[:, :, 1, 0]))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz J10 voltage mag ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_J10_voltage_mag_' + str(freq / 1e6) + 'MHz_ZA' + str(za) + '.png')
    plt.clf()

    plt.imshow(numpy.abs(j[:, :, 1, 1]))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz J11 voltage mag ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_J11_voltage_mag_' + str(freq / 1e6) + 'MHz_ZA' + str(za) + '.png')
    plt.clf()


def plotVisResponse(j, freq, za):
    """
    Utility to plot the visibilty XX,YY,XY and YX response of the array for
    an unpolarised 1Jy source
    Input: j a visibility matrix (complex) of dimensions [za][az][2][2]
    """
    import matplotlib.pyplot as plt

    vis = makeUnpolInstrumentalResponse(j, j)
    plt.imshow(numpy.abs(vis[:, :, 0, 0]))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz XX mag ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_XX_mag_' + str(freq / 1e6) + 'MHz_ZA=' + str(za) + '.png')
    plt.clf()

    plt.imshow(numpy.abs(vis[:, :, 1, 1]))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz YY mag ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_YY_mag_' + str(freq / 1e6) + 'MHz_ZA=' + str(za) + '.png')
    plt.clf()

    plt.imshow(numpy.abs(vis[:, :, 0, 1]))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz XY mag ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_XY_mag_' + str(freq / 1e6) + 'MHz_ZA=' + str(za) + '.png')
    plt.clf()

    plt.imshow(numpy.abs(vis[:, :, 1, 0]))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz YX mag ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_YX_mag_' + str(freq / 1e6) + 'MHz_ZA=' + str(za) + '.png')
    plt.clf()


def plotArrayFactors(ax, ay, za):
    """
    Utility to make a plot of an array factor for debugging
    """
    import matplotlib.pyplot as plt

    plt.imshow(numpy.abs(ax))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz X voltage mag array factor. ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_array_voltage_X_mag_' + str(freq / 1e6) + 'MHz_ZA' + str(za) + '.png')
    plt.clf()

    plt.imshow(numpy.angle(ax))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz X voltage phase array factor ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_array_voltage_X_ph_' + str(freq / 1e6) + 'MHz_ZA' + str(za) + '.png')
    plt.clf()

    plt.imshow(numpy.abs(ay))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz Y voltage mag array factor ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_array_voltage_Y_mag_' + str(freq / 1e6) + 'MHz_ZA' + str(za) + '.png')
    plt.clf()

    plt.imshow(numpy.angle(ay))
    plt.title('MWA ' + str(freq / 1e6) + 'MHz Y voltage phase array factor ZA=' + str(za))
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.savefig('/tmp/MWA_array_voltage_Y_ph_' + str(freq / 1e6) + 'MHz_ZA' + str(za) + '.png')
    plt.clf()


# execute a series of tests if invoked from the command line
if __name__ == "__main__":

    logger.setLevel(logging.DEBUG)
    doplots = True
    lat = config.MWAPOS.lat.deg
    freq = 216e6
    (az, za) = makeAZZA()

    # parallactic angle
    (ha, dec) = h2e(az, za, lat)
    pa = calcParallacticAngle(ha, dec, lat)

    d = Dipole('lookup')
    #    d = Dipole()
    if doplots:
        plotDipoleJones(d, freq)

    # tests with full tile
    # az=0, za=14 degs
    delays1 = numpy.array([[6, 6, 6, 6,
                            4, 4, 4, 4,
                            2, 2, 2, 2,
                            0, 0, 0, 0],
                           [6, 6, 6, 6,
                            4, 4, 4, 4,
                            2, 2, 2, 2,
                            0, 0, 0, 0]],
                          dtype=numpy.float32)
    za_delays = {'0': delays1 * 0, '14': delays1, '28': delays1 * 2}

    tile = ApertureArray(dipoles=[d] * 16)
    if doplots:
        for za_delay in za_delays:
            logger.debug("ZA is: %s. Delays are: %r" % (za_delay, za_delays[za_delay]))
            (ax0, ay0) = tile.getArrayFactor(az, za, freq, za_delays[za_delay])
            logger.info("plotting Array factor voltage for ZA " + str(za_delay))
            plotArrayFactors(ax0, ay0, za_delay)
            logger.info("plotting tile Jones response for ZA " + str(za_delay))
            j = tile.getResponse(az, za, freq, za_delays[za_delay])
            plotArrayJones(j, freq, za_delay)
            logger.info("Plotting visbility response for two identical tiles ZA " + str(za_delay))
            plotVisResponse(j, freq, za_delay)
