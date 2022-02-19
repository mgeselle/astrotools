#!/usr/bin/python3
# Computes the CCD gain and readout noise by Janesick's method.
#
import getopt
import numpy as np
import random
import sys
from astropy.io import fits
from math import sqrt
from pathlib import Path


def usage_exit():
    print('Usage: {} -b bias_dir -f flat_dir'.format(sys.argv[0]))
    sys.exit(1)


def pick_pair(p_in_dir):
    in_files = [x for x in p_in_dir.iterdir() if x.is_file() and x.suffix in ('.fit', '.fits')]
    i1 = random.randint(0, len(in_files))
    i2 = i1
    while i2 == i1:
        i2 = random.randint(0, len(in_files))
    sum_avg = 0
    delta = None
    for i in (i1, i2):
        in_hdu = fits.open(in_files[i])
        data = in_hdu[0].data
        data = data * 1.0
        avg = np.mean(data)
        print('... {}: avg = {:f}'.format(in_files[i], avg))
        sum_avg = sum_avg + avg
        if delta is None:
            delta = data
        else:
            delta = delta - data
        in_hdu.close()
    result = dict()
    result['sum_avg'] = sum_avg
    result['delta_var'] = np.var(delta)
    return result


a_bias_dir = None
a_flat_dir = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'b:f:')
    for opt, arg in opts:
        if opt == '-b':
            a_bias_dir = Path(arg)
        elif opt == '-f':
            a_flat_dir = Path(arg)
        else:
            print('Unsupported option: {}'.format(opt))
            usage_exit()

    if a_bias_dir is None:
        print('Bias frame directory not specified.')
        usage_exit()

    if not a_bias_dir.is_dir():
        print('{} is not a directory or does not exist.'.format(a_bias_dir))
        usage_exit()

    if a_flat_dir is None:
        print('Flat frame directory not specified.')
        usage_exit()

    if not a_flat_dir.is_dir():
        print('{} is nor a directory or does not exist.'.format(a_flat_dir))
        usage_exit()

    print('Flat frames')
    flat = pick_pair(a_flat_dir)
    print('Bias frames')
    bias = pick_pair(a_bias_dir)
    print()

    gain = (flat['sum_avg'] - bias['sum_avg']) / (flat['delta_var'] - bias['delta_var'])
    read_noise = gain * sqrt(bias['delta_var'] / 2)

    print('Gain: {:4.3f} e-/ADU'.format(gain))
    print('Read-out noise: {:5.3f} e-'.format(read_noise))

except getopt.GetoptError as err:
    print(err)
    usage_exit()
