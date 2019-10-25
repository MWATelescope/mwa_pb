import os

import numpy

from scipy import interpolate

from . import config


##################################

def delayset2delaylines(delayset):
    """
       Calculate an array of 5 delay line flags based on the delay setting.
       NOT VECTOR.
    """
    # See if we have a valid delay setting
    if delayset < 0 or delayset > 63:
        raise ValueError("Invalid Delay Setting %s" % repr(delayset))

    # Delay settings with the MSB set are turned off, return None
    if delayset > 31:
        return None

    # Iterate through delaylines
    t = delayset
    dlines = [False] * 5
    for i in range(4, -1, -1):
        if t >= 2 ** i:
            t = t - 2 ** i
            dlines[i] = True

    return dlines


##################################

def get_delay_length(delayset, freq, delayfile=None):
    """
       Get a delay length (in seconds) from a delay set
    """
    if delayfile is None:
        delayfile = config.MEAS_DELAYS

    if not os.path.exists(delayfile):
        raise ValueError("Delay File %s does not exist" % delayfile)

    # Read in the array from the delay file
    t = []
    f = open(delayfile)
    for line in f:
        t.append(list(map(float, line.split())))
    darr = numpy.array(t)

    f_freqs = darr[:, 0]  # Array of frequencies in the delay file.
    # Columns 1 through 5 in the delay file correspond to those delay lines

    delayset = numpy.array(delayset)
    outdelay = numpy.zeros(delayset.size, dtype='float64')

    for j in range(delayset.size):
        # stupid numpy zero-length array type....
        if delayset.size == 1:
            dlines = delayset2delaylines(delayset)
        else:
            dlines = delayset2delaylines(delayset[j])
        for i in range(5):
            # Check if each delay line is on.  If it is, interpolate the file
            # to find the amount of delay to add
            if dlines[i]:
                ifunc = interpolate.splrep(f_freqs, darr[:, i + 1], s=0)
                outdelay[j] = outdelay[j] + interpolate.splev(freq, ifunc, der=0)
    return outdelay


##################################

def get_delay_gains(delayset, freq, delayfile=None):
    """
       Get a delay gains (linear scale) from a delay set
    """
    gainfile = ''
    if delayfile is None:
        gainfile = config.MEAS_GAINS

    if not os.path.exists(gainfile):
        raise ValueError("Gain File %s does not exist" % gainfile)

    # Read in the array from the delay file
    t = []
    f = open(gainfile)
    for line in f:
        t.append(list(map(float, line.split())))
    garr = numpy.array(t)

    f_freqs = garr[:, 0]  # Array of frequencies in the delay file.
    # Columns 1 through 5 in the delay file correspond to those delay lines

    delayset = numpy.array(delayset)
    outgain = numpy.zeros(delayset.size, dtype='float64')
    for j in range(delayset.size):
        # stupid numpy zero-length array type....
        if delayset.size == 1:
            dlines = delayset2delaylines(delayset)
        else:
            dlines = delayset2delaylines(delayset[j])

        for i in range(5):
            # Check if each delay line is on.  If it is, interpolate the file
            # to find the amount of delay to add
            if dlines[i]:
                ifunc = interpolate.splrep(f_freqs, garr[:, i + 1], s=0)
                outgain[j] = outgain[j] + interpolate.splev(freq, ifunc, der=0)

    return (10.0 ** (outgain / 20.0))
