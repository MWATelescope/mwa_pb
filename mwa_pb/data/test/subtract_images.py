#!/usr/bin/env python

import astropy.io.fits as pyfits
# import pylab
import math
# from array import *
# import matplotlib.pyplot as plt
import numpy as np
# import string
import sys
import os
import errno
import getopt

# global parameters :
debug=0
fitsname="file.fits"
fitsname2="fits2.fits"
oper="+"
out_fitsname="sum.fits"
do_show_plots=0
do_gif=0

center_x=1025
center_y=1025
radius=600

def mkdir_p(path):
   try:
      os.makedirs(path)
   except OSError as exc: # Python >2.5
      if exc.errno == errno.EEXIST:
         pass
      else: raise
                                            
def usage():
   print "calcfits.py FITS_FILE1 OPERATION[*,-,compare,a,/,+] FITS_FILE2 OUTPUT_FITS_FILE[default %s]" % out_fitsname
   print "\n"
   print "-d : increases verbose level"
   print "-h : prints help and exists"
   print "-g : produce gif of (channel-avg) for all integrations"

# functions :
def parse_command_line():
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvdg", ["help", "verb", "debug", "gif"])
    except getopt.GetoptError, err:
        # print help information and exit:
        print str(err) # will print something like "option -a not recognized"
        usage()
        sys.exit(2)

    for o, a in opts:
        if o in ("-d","--debug"):
            debug += 1
        if o in ("-v","--verb"):
            debug += 1
        if o in ("-g","--gif"):
            do_gif = 1
        elif o in ("-h", "--help"):
            usage()
            sys.exit()
        else:
            assert False, "unhandled option"
    # ...c

# 
if len(sys.argv) > 1:
   fitsname = sys.argv[1]

if len(sys.argv) > 2:
   fitsname2 = sys.argv[2]

if len(sys.argv) > 3:
   out_fitsname = sys.argv[3]
   
oper="-"
if len(sys.argv) > 4:
   oper = sys.argv[4]


print "####################################################"
print "PARAMTERS :"
print "####################################################"
print "fitsname        = %s" % fitsname
print "oper            = %s" % oper
print "fitsname2       = %s" % fitsname2
print "out_fitsname   = %s" % out_fitsname
print "####################################################"

fits = pyfits.open(fitsname)
x_size=fits[0].header['NAXIS1']
# channels=100
y_size=fits[0].header['NAXIS2']
print 'Read fits file %s' % fitsname
print 'FITS size = %d x %d' % (x_size,y_size)

fits2 = pyfits.open(fitsname2)
x_size2=fits2[0].header['NAXIS1']
# channels=100
y_size2=fits2[0].header['NAXIS2']
print 'Read fits file 2 %s' % fitsname2
print 'FITS size 2 = %d x %d' % (x_size,y_size)

if x_size!=x_size2 or y_size!=y_size2 :
   print "ERROR : cannot execute operation %s on files of different sizes (%d,%d) != (%d,%d)" % (oper,x_size,y_size,x_size2,y_size2)
   exit;

data1=None
if fits[0].data.ndim >= 4 :
   data1=fits[0].data[0][0]
else :
   data1=fits[0].data
   
data2=None
if fits2[0].data.ndim >= 4 :   
   data2=fits2[0].data[0][0]
else :
   data2=fits2[0].data

# print 'BEFORE (%d,%d) = %.2f' % (x_size/2,y_size/2,data[y_size/2][x_size/2])

hdu_out = pyfits.PrimaryHDU()
hdu_out.header = fits[0].header
# hdu_out = fits.PrimaryHDU()
hdu_out.data = np.random.random((x_size,y_size))
data_out=hdu_out.data

diff_count=0

for y in range(y_size) :
   for x in range(x_size) :   
      if oper == "*" : 
         data_out[y][x] = data1[y][x] * data2[y][x]
      if oper == "-" : 
         data_out[y][x] = data1[y][x] - data2[y][x]
      if oper == "/" : 
         data_out[y][x] = data1[y][x] / data2[y][x]
      if oper == "+" : 
         data_out[y][x] = data1[y][x] + data2[y][x]
      if oper == "a" : 
         data_out[y][x] = (data1[y][x] + data2[y][x])/2.00
      if oper == "compare" :
         if math.fabs(data1[y][x] - data2[y][x]) > 0.001 :
            print "Files differ at (x,y) = (%d,%d) %.4f != %.4f" % (x,y,data1[y][x],data2[y][x])


           
hdulist = pyfits.HDUList([hdu_out])
hdulist.writeto(out_fitsname,clobber=True)

if oper == "compare" :
   if diff_count > 0  :
      print "Number of differences = %d" % (diff_count)
   else :
      print "Files are the same : images are the same and headers might still differ"
else :        
   # fits.writeto( out_fitsname , clobber=True )      
   # print 'AFTER (%d,%d) = %.2f' % (x_size/2,y_size/2,data[y_size/2][x_size/2])
   print "Resulting image saved to file %s" % out_fitsname
   

       

