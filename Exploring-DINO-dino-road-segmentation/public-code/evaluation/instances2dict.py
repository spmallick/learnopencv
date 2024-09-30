#!/usr/bin/python
#
# Convert instances from png files to a dictionary
#

from __future__ import print_function
import os, sys

# Cityscapes imports
from instance import *
sys.path.append( os.path.normpath( os.path.join( os.path.dirname( __file__ ) , '..' , 'helpers' ) ) )
# from csHelpers import *
from PIL import Image
import numpy as np
from anue_labels import *

def instances2dict(imageFileList, verbose=False):
    imgCount     = 0
    instanceDict = {}

    if not isinstance(imageFileList, list):
        imageFileList = [imageFileList]

    if verbose:
        print("Processing {} images...".format(len(imageFileList)))

    for imageFileName in imageFileList:
        # Load image
        img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            if instanceId != 255 and instanceId//1000 <19 and instanceId//1000>5:
                instanceObj = Instance(imgNp, instanceId)

                instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

        if verbose:
            print("\rImages Processed: {}".format(imgCount), end=' ')
            sys.stdout.flush()

    if verbose:
        print("")

    return instanceDict

def main(argv):
    fileList = []
    if (len(argv) > 2):
        for arg in argv:
            if ("png" in arg):
                fileList.append(arg)
    instances2dict(fileList, True)

if __name__ == "__main__":
    main(sys.argv[1:])
