#!/usr/bin/python3
#
# Combines a number of images into a master frame.
#
import astropy.io.fits as fits
import getopt
import numpy as np
import re
import sys
from pathlib import Path


def usage_exit():
    print('Usage: {} -i input_dir -o output [-p pattern] [-a] [-c] [-s]'.format(sys.argv[0]))
    sys.exit(1)


# MAIN Main main
try:
    opts, args = getopt.getopt(sys.argv[1:], 'aci:o:p:s')

    in_dir = None
    out_path = None
    pattern = re.compile(r'.*\.fits?')
    combine_fun = np.median
    do_scale = False
    do_clip = False
    for opt, arg in opts:
        print(opt)
        if opt == '-a':
            combine_fun = np.mean
        if opt == '-c':
            do_clip = True
        elif opt == '-i':
            in_dir = Path(arg)
        elif opt == '-o':
            out_path = Path(arg)
        elif opt == '-p':
            pattern = re.compile(arg)
        elif opt == '-s':
            do_scale = True

    if in_dir is None:
        print('Input directory not specified.')
        usage_exit()
    if out_path is None:
        print('Output file not specified.')
        usage_exit()
    if not in_dir.is_dir():
        print('Not a directory: {}'.format(in_dir))
        usage_exit()

    in_files = [f for f in in_dir.iterdir() if f.is_file() and pattern.fullmatch(f.name)]
    in_files.sort()
    if not in_files:
        print('No input files found.')
        sys.exit(1)

    header = None
    data = []

    for file in in_files:
        hdu_list = fits.open(file)
        if header is None:
            header = hdu_list[0].header
            header['HISTORY'] = 'Created master frame from {:d} input frames.'.format(len(in_files))
        file_data = hdu_list[0].data
        if do_scale:
            median = np.median(file_data)
            # Avoid issues when dividing by scaled frame later on.
            file_data = file_data.clip(0.3 * median)
            file_data = file_data / median
            header['HISTORY'] = '... {}, scaled by {:8.1f}'.format(file.name, median)
        else:
            header['HISTORY'] = '... {}'.format(file.name)
        data.append(file_data)
        hdu_list.close()
    out_data = combine_fun(data, axis=0)
    if do_clip:
        ozt_data = out_data.clip(0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(out_path, out_data, header, overwrite=True)

except getopt.GetoptError as err:
    print(err)
    usage_exit()
