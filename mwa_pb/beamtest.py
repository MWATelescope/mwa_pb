
import math

import numpy

from matplotlib import pyplot

import mwapb


def plot_beam(delays=numpy.zeros(16), gains=numpy.ones(16), stokes='I'):
    t = numpy.mgrid[0:91,0:361]
    el = t[0, :, :]
    az = t[1, :, :]
    
    dtor = math.pi / 180.0
    theta = (90 - el) * dtor
    phi = az * dtor
    ll = numpy.sin(theta) * numpy.sin(phi)
    m = numpy.sin(theta) * numpy.cos(phi)

    tbeam = mwapb.MWA_tile_gain(freq=300e6, stokes=stokes, delays=delays, gains=gains)
    pyplot.contourf(ll, m, tbeam.calculate(az, el), 256)
    pyplot.draw()


if __name__ == "__main__":
    pyplot.ion()
    plot_beam()
    pyplot.show()
