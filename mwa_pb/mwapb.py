import logging
import math

import numpy

from . import measured_beamformer

logging.basicConfig(format='# %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger(__name__)  # default logger level is WARNING

logger.warning("This module '" + __name__ + "' is deprecated and does not use the up-to-date beam models. Use the new primary beam model instead in mwa_tile.")


#########################

class voltage_pattern(object):
    """
      Base class for a complex voltage pattern.  Has a calculate method, which gets a pattern for a theta/phi value
      theta=zenith angle in radians
      phi=azimuth in radians, with north=0, east=pi/2
    """
    freq = None

    def set_freq(self, freq):
        if freq < 1.0E6 or freq > 1.0E9:
            logging.warning(
                'Frequency is out of the low-frequency radio band... are you sure you meant this frequency?')
        self.freq = freq

    def calculate(self, theta, phi, pol=None):
        logging.warning('This is only a dummy class!  Nothing calculated.')

    def set_pol(self, pol):
        pass


#########################


class isotropic_dipole_vpat(voltage_pattern):
    """
      Class for a purely isotropic dipole
    """

    def calculate(self, theta, phi, pol=None):
        return numpy.ones_like(theta)


#########################

class short_dipole_vpat(voltage_pattern):
    """
      Class for a short-dipole over a ground screen complex voltage pattern
    """

    def __init__(self, pol=None):
        self.height = 0.278
        self.freq = 100.0E6
        if pol is not None:
            self.set_pol(pol)
        else:
            self.daz = 0

    def set_pol(self, pol):
        if pol == 'Y':
            self.daz = 0
        elif pol == 'X':
            self.daz = 90
        else:
            raise ValueError("Only valid polarizatons are 'X' or 'Y'")

    def calculate(self, theta, phi, pol=None):
        """
          Calculates the complex voltage pattern of an electrically short dipole over a ground screen

          dip=short_dipole()
          print short_dipole.calculate(theta,phi)

          theta=zenith angle in radians
          phi=azimuth in radians, with north=0, east=pi/2
        """
        lam = 2.998e8 / self.freq
        phi = phi + self.daz * math.pi / 180.0
        dphase = 4 * math.pi * self.height * numpy.cos(theta) / lam

        # Power pattern of an isolated short dipole goes like: 1-(cosphi**2)*(sintheta**2)
        # Complex voltage pattern of an isotropic dopole over a ground plane goes like (1-numpy.exp(1j*dphase))
        # Be careful with what is a total power pattern (square magnitude!) and a complex voltage

        ret = numpy.sqrt(1 - (numpy.cos(phi) ** 2) * (numpy.sin(theta) ** 2)) * (1 - numpy.exp(1j * dphase))

        # No response below horizon
        ret.put(0.0, numpy.array(theta > (math.pi / 2)).nonzero())
        return ret


#########################

class MWA_tile_vpat(object):
    """
      Class for an MWA tile pattern.  By default it uses a short dipole as the element pattern.
      You can specify individual element patterns or gains by giving a 16-element array of these quantities.
    """

    def __init__(self,
                 freq=100.0E6,
                 delays=0.0,
                 gains=1.0,
                 gain_func=None,
                 delay_func=None,
                 element_patterns=None,
                 pol='Y'):

        if element_patterns is None:
            element_patterns = short_dipole_vpat()

        self.freq = None
        self.delays = None
        self.gains = None
        self.element_patterns = None

        self.set_gains(gains)
        self.set_delays(delays)
        self.set_element_patterns(element_patterns)
        self.set_pol(pol)
        self.set_freq(freq)

        if delay_func is None:
            self.delay_func = lambda x, f: x * 435E-12
        else:
            self.delay_func = delay_func

        if gain_func is None:
            self.gain_func = lambda x, f: 1.0
        else:
            self.gain_func = gain_func

        dip_sep = 1.1
        self.dipole_y = dip_sep * numpy.array([1.5, 1.5, 1.5, 1.5,
                                               0.5, 0.5, 0.5, 0.5,
                                               -0.5, -0.5, -0.5, -0.5,
                                               -1.5, -1.5, -1.5, -1.5])
        self.dipole_x = dip_sep * numpy.array([-1.5, -0.5, 0.5, 1.5,
                                               -1.5, -0.5, 0.5, 1.5,
                                               -1.5, -0.5, 0.5, 1.5,
                                               -1.5, -0.5, 0.5, 1.5])

    def set_pol(self, pol):
        for pat in self.element_patterns:
            pat.set_pol(pol)

    def set_freq(self, freq):
        self.freq = freq
        for pat in self.element_patterns:
            pat.set_freq(freq)

    def set_delays(self, delays):
        delays = numpy.array(delays)
        if delays.size == 1:
            delays = numpy.tile(delays, 16)
        self.delays = delays
        if self.delays.size != 16:
            raise ValueError('delays must be a 16-element array')

    def set_gains(self, gains):
        gains = numpy.array(gains)
        if gains.size == 1:
            gains = numpy.tile(gains, 16)
        self.gains = gains
        if self.gains.size != 16:
            raise ValueError('gains must be a 16-element array')

    def set_element_patterns(self, element_patterns):
        element_patterns = numpy.array(element_patterns)
        if element_patterns.size == 1:
            element_patterns = numpy.tile(element_patterns, 16)
        self.element_patterns = element_patterns
        if self.element_patterns.size != 16:
            raise ValueError('element_patterns must be a 16-element array of functions')

    def calculate(self, theta, phi):
        lam = 2.998e8 / self.freq
        dy = self.dipole_y
        dx = self.dipole_x

        dphase = 2 * math.pi * self.freq * (self.delay_func(self.delays, self.freq))  # phase delay (in radians) at each tile

        # for i=0,n_elements(dip_gains)-1 do dip_gains[i]=dip_gains[i]*get_delay_gains(delays[i],freq)
        ff = 2 * math.pi * 1j / lam
        resp = numpy.zeros_like(theta + phi + 1j)

        for dipole in range(16):
            resp = (resp +
                    self.gain_func(self.delays[dipole], self.freq) *
                    self.gains[dipole] *
                    self.element_patterns[dipole].calculate(theta, phi) *
                    numpy.exp(ff * dy[dipole] * numpy.sin(theta) * numpy.cos(phi) +
                              ff * dx[dipole] * numpy.sin(theta) * numpy.sin(phi) + -1j * dphase[dipole]))
        return resp


######################################################################


class gain_pattern(object):

    def __init__(self):
        self.freq = None
        self.vpat = None
        self._stokes = 'I'
        self.norm_az = 0.0
        self.norm_el = 90.0
        self.normalize = True

    def set_stokes(self, stokes):
        goodstokes = ['XX', 'YY', 'XY', 'YX', 'I', 'Q', 'U', 'V']
        if stokes in goodstokes:
            self._stokes = stokes

    def set_freq(self, freq):
        self.freq = freq

    def get_stokes(self):
        return self._stokes

    def calculate(self, az, el):
        return None


######################################################################

class MWA_tile_gain(gain_pattern):

    def __init__(self, freq=100.0E6, stokes='I', delays=None, gains=None, idealbf=False):
        gain_pattern.__init__(self)

        self.set_freq(freq)
        self.set_stokes(stokes)
        self.vpat = MWA_tile_vpat()
        self.nvpat = MWA_tile_vpat()
        if gains is not None:
            self.vpat.set_gains(gains)
            self.nvpat.set_gains(gains)
        if delays is not None:
            self.vpat.set_delays(delays)

        if idealbf is False:
            self.vpat.gain_func = measured_beamformer.get_delay_gains
            self.vpat.delay_func = measured_beamformer.get_delay_length
            self.nvpat.gain_func = measured_beamformer.get_delay_gains
            self.nvpat.delay_func = measured_beamformer.get_delay_length

    # Changed normalisation to peak of zenith beam
    # by adding a voltage pattern set to zenith (nvpat) and dividing
    # by its maximum for each polarisation - DJ&NHW 15/05/2012

    def set_delays(self, delays):
        self.vpat.set_delays(delays)

    def set_gains(self, gains):
        self.vpat.set_gains(gains)

    def calculate(self, az, el):
        # print 'at entrance to calculate, self._stokes='+str(self._stokes)
        dtor = math.pi / 180.0
        theta = (90 - el) * dtor
        phi = az * dtor

        self.nvpat.set_freq(self.freq)
        self.vpat.set_freq(self.freq)
        if self._stokes == 'XX':
            self.vpat.set_pol('X')
            self.nvpat.set_pol('X')
            beam = self.vpat.calculate(theta, phi)
            if self.normalize is True:
                n = self.nvpat.calculate(self.norm_az, self.norm_el)
                beam = beam / n
            sqbeam = numpy.abs(beam) ** 2
            return sqbeam
        elif self._stokes == 'YY':
            self.vpat.set_pol('Y')
            self.nvpat.set_pol('Y')
            beam = self.vpat.calculate(theta, phi)
            if self.normalize is True:
                n = self.nvpat.calculate(self.norm_az, self.norm_el)
                beam = beam / n
            sqbeam = numpy.abs(beam) ** 2
            return sqbeam
        else:
            self.vpat.set_pol('X')
            self.nvpat.set_pol('X')
            bx = self.vpat.calculate(theta, phi)
            if self.normalize is True:
                n = self.nvpat.calculate(self.norm_az, self.norm_el)
                bx = bx / n
            self.vpat.set_pol('Y')
            self.nvpat.set_pol('Y')
            by = self.vpat.calculate(theta, phi)
            if self.normalize is True:
                n = self.nvpat.calculate(self.norm_az, self.norm_el)
                by = by / n
            if self._stokes == 'XY':
                return bx * numpy.conj(by)
            elif self._stokes == 'YX':
                return by * numpy.conj(bx)
            elif self._stokes == 'I':
                return (numpy.abs(bx) ** 2 + numpy.abs(by) ** 2) / 2.0
            elif self._stokes == 'Q':
                return (numpy.abs(bx) ** 2 - numpy.abs(by) ** 2) / 2.0
            elif self._stokes == 'U':
                return numpy.real(bx * numpy.conj(by))
            elif self._stokes == 'V':
                return numpy.imag(bx * numpy.conj(by))
            else:
                raise ValueError('Invalid Stokes!')
