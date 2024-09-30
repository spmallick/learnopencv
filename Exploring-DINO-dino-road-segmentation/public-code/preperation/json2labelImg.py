#!/usr/bin/python
#

# python imports
import sys
sys.path.append("/home/somusan/OpencvUni/opencvblog/robotics-series/yolop_idd/public-code/helpers")
from anue_labels import labels, name2label
from annotation import Annotation
import os
import sys
import getopt

import numpy

# Image processing
# Check if PIL is actually Pillow as expected
# try:
#     from PIL import PILLOW_VERSION
# except:
#     print("Please install the module 'Pillow' for image processing, e.g.")
#     print("pip install pillow")
#     sys.exit(-1)

# try:
#     import PIL.Image as Image
#     import PIL.ImageDraw as ImageDraw
# except:
#     print("Failed to import the image processing packages.")
#     sys.exit(-1)

from PIL import Image, ImageDraw


sys.path.append(os.path.normpath(os.path.join(
    os.path.dirname(__file__), '..', 'helpers')))

# Print the information


def printHelp():
    print('{} [OPTIONS] inputJson outputImg'.format(
        os.path.basename(sys.argv[0])))
    print('')
    print('Reads labels as polygons in JSON format and converts them to label images,')
    print('where each pixel has an ID that represents the ground truth label.')
    print('')
    print('Options:')
    print(' -h                 Print this help')
    print(' -t                 Use the "trainIDs" instead of the regular mapping. See "labels.py" for details.')

# Print an error message and quit


def printError(message):
    print('ERROR: {}'.format(message))
    print('')
    print('USAGE:')
    printHelp()
    sys.exit(-1)

# Convert the given annotation to a label image


def createLabelImage(inJson, annotation, encoding, outline=None):
    # the size of the image
    size = (annotation.imgWidth, annotation.imgHeight)

    # the background
    if encoding == "id":
        background = name2label['unlabeled'].id
    elif encoding == "csId":
        background = name2label['unlabeled'].csId
    elif encoding == "csTrainId":
        background = name2label['unlabeled'].csTrainId
    elif encoding == "level4Id":
        background = name2label['unlabeled'].level4Id
    elif encoding == "level3Id":
        background = name2label['unlabeled'].level3Id
    elif encoding == "level2Id":
        background = name2label['unlabeled'].level2Id
    elif encoding == "level1Id":
        background = name2label['unlabeled'].level1Id
    elif encoding == "color":
        background = name2label['unlabeled'].color
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    if encoding == "color":
        labelImg = Image.new("RGBA", size, background)
    else:
        # print(size, background)
        labelImg = Image.new("L", size, background)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw(labelImg)

    # loop over all objects
    for obj in annotation.objects:
        
        label = obj.label
        polygon = obj.polygon

        if label == 'road':
            # If the object is deleted, skip it
            if obj.deleted or len(polygon) < 3:
                continue

            # If the label is not known, but ends with a 'group' (e.g. cargroup)
            # try to remove the s and see if that works
            if (not label in name2label) and label.endswith('group'):
                label = label[:-len('group')]

            if not label in name2label:
                print("Label '{}' not known.".format(label))
                tqdm.write("Something wrong in: " + inJson)
                continue

            # If the ID is negative that polygon should not be drawn
            if name2label[label].id < 0:
                continue

            if encoding == "id":
                val = name2label[label].id
            elif encoding == "csId":
                val = name2label[label].csId
            elif encoding == "csTrainId":
                val = name2label[label].csTrainId
            elif encoding == "level4Id":
                val = name2label[label].level4Id
            elif encoding == "level3Id":
                val = name2label[label].level3Id
            elif encoding == "level2Id":
                val = name2label[label].level2Id
            elif encoding == "level1Id":
                val = name2label[label].level1Id
            elif encoding == "color":
                val = name2label[label].color

            try:
                if outline:

                    drawer.polygon(polygon, fill=val, outline=outline)
                else:
                    drawer.polygon(polygon, fill=val)
                    # print(label, val)
            except:
                print("Failed to draw polygon with label {}".format(label))
                raise

    # print(numpy.array(labelImg))

    return labelImg

# A method that does all the work
# inJson is the filename of the json file
# outImg is the filename of the label image that is generated
# encoding can be set to
#     - "ids"      : classes are encoded using the regular label IDs
#     - "trainIds" : classes are encoded using the training IDs
#     - "color"    : classes are encoded using the corresponding colors


def json2labelImg(inJson, outImg, encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(inJson)
    labelImg = createLabelImage(inJson, annotation, encoding)
    labelImg.save(outImg)

# The main method, if you execute this script directly
# Reads the command line arguments and calls the method 'json2labelImg'


def main(argv):
    trainIds = False
    try:
        opts, args = getopt.getopt(argv, "ht")
    except getopt.GetoptError:
        printError('Invalid arguments')
    for opt, arg in opts:
        if opt == '-h':
            printHelp()
            sys.exit(0)
        elif opt == '-t':
            trainIds = True
        else:
            printError("Handling of argument '{}' not implementend".format(opt))

    if len(args) == 0:
        printError("Missing input json file")
    elif len(args) == 1:
        printError("Missing output image filename")
    elif len(args) > 2:
        printError("Too many arguments")

    inJson = args[0]
    outImg = args[1]

    if trainIds:
        json2labelImg(inJson, outImg, "trainIds")
    else:
        json2labelImg(inJson, outImg)


# call the main method
if __name__ == "__main__":
    main(sys.argv[1:])
