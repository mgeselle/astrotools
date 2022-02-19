#!/usr/bin/python3
# Converts an input FITS file to a 16 bit FITS file,
# optionally using signed 16 bit numbers. FITS files using
# signed numbers are required for V Spec.
import getopt
import sys
from pathlib import Path
from astropy.io import fits
import numpy as np


def usage():
    print("Usage: {} [-u] [-c clip_max] -i input -o output".format(sys.argv[0]))


try:
    opts, args = getopt.getopt(sys.argv[1:], 'c:i:o:u')
except getopt.GetoptError as err:
    print(err)
    usage()
    sys.exit(1)

inputFile = None
outputFile = None
signed = False
upperLimit = 2**16 - 1
lowerLimit = 0
clipMax = 0

for opt, arg in opts:
    if opt == '-i':
        inputFile = Path(arg)
    elif opt == '-o':
        outputFile = Path(arg)
    elif opt == '-u':
        lowerLimit = -(2**15)
        upperLimit = -lowerLimit - 1
    elif opt == '-c':
        clipMax = int(arg)
    else:
        print("Unsupported option {}".format(opt))
        usage()
        exit(1)

if inputFile is None:
    print('Input file not specified.')
    usage()
    sys.exit(1)

if outputFile is None:
    print('Output file not specified.')
    usage()
    sys.exit(1)

if not inputFile.exists():
    print("Input file not found.")
    exit(1)

inputHduList = fits.open(inputFile)
inputData = inputHduList[0].data
header = inputHduList[0].header
minInput = np.min(inputData)
if clipMax > 0:
    inputData = inputData.clip(minInput, clipMax)
maxInput = np.max(inputData)
print('Max: {}, min: {}'.format(maxInput, minInput))
print('Lower limit: {}, upper limit: {}'.format(lowerLimit, upperLimit))

scale = upperLimit / maxInput
if minInput < lowerLimit:
    if lowerLimit < 0:
        scale = lowerLimit / minInput
    else:
        inputData = inputData - minInput
        maxInput = maxInput - minInput
        scale = upperLimit / maxInput

if scale * maxInput > upperLimit:
    scale = upperLimit / maxInput

inputData = inputData * scale
outputData = None
if lowerLimit < 0:
    outputData = inputData.astype(np.int16)
else:
    outputData = inputData.astype(np.uint16)

header['HISTORY'] = 'Scaled by {:6.4f} and converted to 16bit'.format(scale)

outHdu = fits.PrimaryHDU(outputData, header)
outHdu.writeto(outputFile, overwrite=True)
