#!/usr/bin/python
#
# Various helper methods and includes for Cityscapes
#

# Python imports
import os, sys, getopt
import glob
import math
import json
from collections import namedtuple

# Image processing
# Check if PIL is actually Pillow as expected
# try:
#     from PIL import PILLOW_VERSION
# except:
#     print("Please install the module 'Pillow' for image processing, e.g.")
#     print("pip install pillow")
#     sys.exit(-1)

# try:
#     import PIL.Image     as Image
#     import PIL.ImageDraw as ImageDraw
# except:
#     print("Failed to import the image processing packages.")
#     sys.exit(-1)

from PIL import Image, ImageDraw

# Numpy for datastructures
try:
    import numpy as np
except:
    print("Failed to import numpy package.")
    sys.exit(-1)

# Cityscapes modules
# try:
from annotation   import Annotation
from anue_labels       import labels, name2label, id2label
# except:
#     print("Failed to find all Cityscapes modules")
#     sys.exit(-1)

# Print an error message and quit
def printError(message):
    print('ERROR: ' + str(message))
    sys.exit(-1)

# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val, args):
    if not args.colorized:
        return ""
    if not isinstance(val, float) or math.isnan(val):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN

# Cityscapes files have a typical filename structure
# <city>_<sequenceNb>_<frameNb>_<type>[_<type2>].<ext>
# This class contains the individual elements as members
# For the sequence and frame number, the strings are returned, including leading zeros
CsFile = namedtuple( 'csFile' , [ 'city' , 'sequenceNb' , 'frameNb' , 'type' , 'type2' , 'ext' ] )

# Returns a CsFile object filled from the info in the given filename
def getCsFileInfo(fileName):
    baseName = os.path.basename(fileName)
    parts = baseName.split('_')
    parts = parts[:-1] + parts[-1].split('.')
    if not parts:
        printError( 'Cannot parse given filename ({}). Does not seem to be a valid Cityscapes file.'.format(fileName) )
    if len(parts) == 5:
        csFile = CsFile( *parts[:-1] , type2="" , ext=parts[-1] )
    elif len(parts) == 6:
        csFile = CsFile( *parts )
    else:
        printError( 'Found {} part(s) in given filename ({}). Expected 5 or 6.'.format(len(parts) , fileName) )

    return csFile

# Returns the part of Cityscapes filenames that is common to all data types
# e.g. for city_123456_123456_gtFine_polygons.json returns city_123456_123456
def getCoreImageFileName(filename):
    csFile = getCsFileInfo(filename)
    return "{}_{}_{}".format( csFile.city , csFile.sequenceNb , csFile.frameNb )

# Returns the directory name for the given filename, e.g.
# fileName = "/foo/bar/foobar.txt"
# return value is "bar"
# Not much error checking though
def getDirectory(fileName):
    dirName = os.path.dirname(fileName)
    return os.path.basename(dirName)

# Make sure that the given path exists
def ensurePath(path):
    if not path:
        return
    if not os.path.isdir(path):
        os.makedirs(path)

# Write a dictionary as json file
def writeDict2JSON(dictName, fileName):
    with open(fileName, 'w') as f:
        f.write(json.dumps(dictName, default=lambda o: o.__dict__, sort_keys=True, indent=4))

# dummy main
if __name__ == "__main__":
    printError("Only for include, not executable on its own.")
