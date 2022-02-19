#!/usr/bin/python3
# Reduces a set of images by subtracting bias and dark frames and applying a
# flat frame.
import getopt
import sys
from pathlib import Path
from astropy.io import fits


def usage():
    print("Usage %s -i input -o output [-b bias] [-d dark] [-f flat]" % sys.argv[0])


try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:o:b:d:f:')
except getopt.GetoptError as err:
    print(err)
    usage()
    sys.exit(1)

inputDir = None
outputDir = None
biasFile = None
flatFile = None
darkFile = None

for opt, arg in opts:
    if opt == '-i':
        inputDir = Path(arg)
    elif opt == '-o':
        outputDir = Path(arg)
    elif opt == '-b':
        biasFile = Path(arg)
    elif opt == '-d':
        darkFile = Path(arg)
    elif opt == '-f':
        flatFile = Path(arg)
    else:
        print("Unsupported option %s" % opt)
        usage()
        sys.exit(1)

if not inputDir:
    print("Input directory not specified,")
    usage()
    sys.exit(1)

if not inputDir.exists():
    print("Input directory doesn't exist.")
    sys.exit(1)

if not inputDir.is_dir():
    print("Input directory isn't a directory.")
    sys.exit(1)

if not outputDir:
    print("Output directory not specified")
    sys.exit(1)

biasData = None
biasHduList = None
if biasFile:
    biasHduList = fits.open(biasFile)
    biasData = biasHduList[0].data

darkData = None
darkHduList = None
darkExposure = None
if darkFile:
    darkHduList = fits.open(darkFile)
    darkData = darkHduList[0].data
    darkExposure = float(darkHduList[0].header['EXPTIME'])

flatData = None
flatHduList = None
if flatFile:
    flatHduList = fits.open(flatFile)
    flatData = flatHduList[0].data

outputDir.mkdir(parents=True, exist_ok=True)

for inputFile in inputDir.iterdir():
    if not inputFile.is_file():
        continue
    if inputFile.suffix not in ('.fits', '.fit'):
        continue

    print('Processing {}'.format(inputFile.name))

    inputHduList = fits.open(inputFile)
    inputHeader = inputHduList[0].header
    outputData = inputHduList[0].data

    if biasFile:
        print('Subtracting bias frame {}'.format(biasFile.name))
        inputHeader['HISTORY'] = 'Bias-corrected using {}'.format(biasFile.name)
        outputData = outputData - biasData

    if darkFile:
        exposure = float(inputHeader['EXPTIME'])
        darkScale = exposure / darkExposure
        print('Subtracting dark frame {} scaled by {:6.4f}'.format(darkFile.name, darkScale))
        inputHeader['HISTORY'] = 'Dark-corrected using {} scaled by {:6.4f}'.format(darkFile.name, darkScale)
        outputData = outputData - darkData * darkScale

    if flatFile:
        print('Dividing by flat frame {}'.format(flatFile.name))
        inputHeader['HISTORY'] = 'Flat-corrected using {}'.format(flatFile.name)
        outputData = outputData / flatData

    outputData = outputData.clip(0)
    outHdu = fits.PrimaryHDU(outputData, inputHeader)
    outHdu.writeto(outputDir / inputFile.name)
    print('Output written to {}'.format(outputDir / inputFile.name))

    inputHduList.close()
