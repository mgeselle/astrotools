#!/usr/bin/python3
# Attempts to remove the background from an image.
import getopt
import sys
from pathlib import Path
from astropy.io import fits
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground, MeanBackground, MMMBackground
from photutils.background import ModeEstimatorBackground, SExtractorBackground, BiweightLocationBackground


def usage():
    print("Usage: {} -i input -o output".format(sys.argv[0]))


try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:o:b:')
except getopt.GetoptError as err:
    print(err)
    usage()
    sys.exit(1)

inputFile = None
outputFile = None
bkgSelector = 'd'

for opt, arg in opts:
    if opt == '-i':
        inputFile = Path(arg)
    elif opt == '-o':
        outputFile = Path(arg)
    elif opt == '-b':
        bkgSelector = arg
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

bkgEstimator = None
if bkgSelector == 'd':
    bkgEstimator = MedianBackground()
elif bkgSelector == 'e':
    bkgEstimator = MeanBackground()
elif bkgSelector == 'o':
    bkgEstimator = ModeEstimatorBackground()
elif bkgSelector == 'M':
    bkgEstimator = MMMBackground()
elif bkgSelector == 'S':
    bkgEstimator = SExtractorBackground()
elif bkgSelector == 'B':
    bkgEstimator = BiweightLocationBackground()
else:
    print("Unknown background selector. Defaulting to MedianBackground.")
    bkgEstimator = MedianBackground()

sigmaClip = SigmaClip(sigma=3.)
bkg = Background2D(inputData, (50, 50), filter_size=(3, 3),
                   sigma_clip=sigmaClip, bkg_estimator=bkgEstimator)

outputData = inputData - bkg.background
header['HISTORY'] = 'Removed background using Astropy {}'.format(type(bkgEstimator).__name__)
outHdu = fits.PrimaryHDU(outputData, header)
outHdu.writeto(outputFile, overwrite=True)
print("Output written to {}.".format(outputFile))
