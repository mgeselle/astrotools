#!/usr/bin/python3
#
# Computes and prints statistics of an image.
#
import astropy.io.fits as fits
import numpy as np
import sys


def usage_exit():
    print('Usage: {} file_name'.format(sys.argv[0]))
    sys.exit(1)


# MAIN Main main
if len(sys.argv) != 2:
    usage_exit()

hdul = fits.open(sys.argv[1])

avg = np.mean(hdul[0].data)
median = np.median(hdul[0].data)
min_val = np.min(hdul[0].data)
max_val = np.max(hdul[0].data)
stddev = np.std(hdul[0].data)

hdul.close()

print('Average: {:8.1f}'.format(avg))
print('Median:  {:8.1f}'.format(median))
print('Minimum: {:8.1f}'.format(min_val))
print('Maximum: {:8.1f}'.format(max_val))
print('\u03c3:       {:8.1f}'.format(stddev))
