from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import time
from itertools import combinations
import math

def calculateDistance(dx, dy, dz):
    distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return distance

# Tiny yolo v3/4 label texts
labelMap = [
    "person"
]

syncNN = True

# Get argument first
nnBlobPath = str(Path('models/tiny-yolo-v4_openvino_2021.2_6shave.blob').resolve().absolute())
if len(sys.argv) > 1:
    nnBlobPath = sys.argv[1]

if not Path(nnBlobPath).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - color camera
colorCam = pipeline.createColorCamera()
spatialDetectionNetwork = pipeline.createYoloSpatialDetectionNetwork()
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()

xoutRgb = pipeline.createXLinkOut()
xoutNN = pipeline.createXLinkOut()
xoutBoundingBoxDepthMapping = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")


colorCam.setPreviewSize(416, 416)
colorCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
colorCam.setInterleaved(False)
colorCam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setOutputDepth(True)
stereo.setConfidenceThreshold(255)

spatialDetectionNetwork.setBlobPath(nnBlobPath)
spatialDetectionNetwork.setConfidenceThreshold(0.5)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)
# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(80)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
spatialDetectionNetwork.setIouThreshold(0.5)

# Create outputs

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

colorCam.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    colorCam.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

def createBirdFrame():
    maxZ = 5000
    minZ = 300
    fov = 68.7938
    minDistance = 0.827
    birdFrame = np.zeros((416, 100, 3), np.uint8)
    minY = int((1 - (minDistance - minZ) / (maxZ - minZ)) * birdFrame.shape[0])
    cv2.rectangle(birdFrame, (0, minY), (birdFrame.shape[1], birdFrame.shape[0]), (70, 70, 70), -1)
    alpha = (180 - fov) / 2
    center = int(birdFrame.shape[1] / 2)
    maxP = birdFrame.shape[0] - int(math.tan(math.radians(alpha)) * center)
    fovCnt = np.array([
        (0, birdFrame.shape[0]),
        (birdFrame.shape[1], birdFrame.shape[0]),
        (birdFrame.shape[1], maxP),
        (center, birdFrame.shape[0]),
        (0, maxP),
        (0, birdFrame.shape[0]),
    ])
    birdFrame = cv2.fillPoly(birdFrame, [fovCnt], color=(70, 70, 70))
    return birdFrame

def calcX(val, distanceBirdFrame):
    maxX = 600
    minX = -600
    norm = min(maxX, max(val, minX))
    center = (norm - minX) / (maxX - minX) * distanceBirdFrame.shape[1]
    bottomX = max(center - 2, 0)
    topX = min(center + 2, distanceBirdFrame.shape[1])
    return int(bottomX), int(topX)

def calcZ(val, distanceBirdFrame):
    maxZ = 5000
    minZ = 300
    norm = min(maxZ, max(val, minZ))
    center = (1 - (norm - minZ) / (maxZ - minZ)) * distanceBirdFrame.shape[0]
    bottomZ = max(center - 2, 0)
    topZ = min(center + 2, distanceBirdFrame.shape[0])
    return int(bottomZ), int(topZ)

# Pipeline is defined, now we can connect to the device

with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMapping = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    frame = None
    detections = []

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    while True:
        birdFrame = createBirdFrame()
        inPreview = previewQueue.get()
        inNN = detectionNNQueue.get()
        depth = depthQueue.get()

        counter+=1
        currentTime = time.monotonic()
        if (currentTime - startTime) > 1 :
            fps = counter / (currentTime - startTime)
            counter = 0
            startTime = currentTime

        frame = inPreview.getCvFrame()
        detections = inNN.detections

        if len(detections) != 0:

        # If the frame is available, draw bounding boxes on it and show the frame
            height = frame.shape[0]
            width  = frame.shape[1]
            centroidDict = dict()
            objectId = 0
            for detection in detections:
                try:
                    label = labelMap[detection.label]
                except:
                    label = detection.label
                if str(label) == 'person' :
                    boundingBoxMapping = xoutBoundingBoxDepthMapping.get()
                    roiDatas = boundingBoxMapping.getConfigData()

                    for roiData in roiDatas:
                        roi = roiData.roi
                        #roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                        topLeft = roi.topLeft()
                        bottomRight = roi.bottomRight()
                        xmin = int(topLeft.x)
                        ymin = int(topLeft.y)
                        xmax = int(bottomRight.x)
                        ymax = int(bottomRight.y)

                    # Denormalize bounding box
                        x1 = int(detection.xmin * width)
                        x2 = int(detection.xmax * width)
                        y1 = int(detection.ymin * height)
                        y2 = int(detection.ymax * height)
                        center = (int((x1+x2)/2), int((y1+y2)/2))
                        xsp, ysp, zsp = int(detection.spatialCoordinates.x), int(detection.spatialCoordinates.y), int(detection.spatialCoordinates.z)
                        centroidDict[objectId] = (x1, x2, y1, y2, xsp, ysp, zsp, center)
                        objectId += 1

            redZoneList = [] # List containing which Object id is in under threshold distance condition.

            for (id1, p1), (id2, p2) in combinations(centroidDict.items(), 2):
                dx, dy, dz = p1[4] - p2[4], p1[5] - p2[5], p1[6] - p2[6]
                distance = calculateDistance(dx, dy, dz)
                if(int(distance) < 2000 and int(distance) != 0):
                    startPoint = p1[7]
                    endPoint = p2[7]
                    textCoord = (int((startPoint[0]+endPoint[0])/2),int((startPoint[1]+endPoint[1])/2)+20)
                    cv2.line(frame, startPoint, endPoint, (0, 0, 255), 2)
                    cv2.putText(frame, str(round(distance/1000,2))+' m', textCoord, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if id1 not in redZoneList:
                        redZoneList.append(id1)
                    if id2 not in redZoneList:
                        redZoneList.append(id2)

            for idx, box in centroidDict.items():
                left, right = calcX(box[4], birdFrame)
                top, bottom = calcZ(box[6], birdFrame)
                if idx in redZoneList:   # if id is in red zone list
                    cv2.rectangle(frame, (box[0], box[2]), (box[1], box[3]), (0, 0, 255), 2) # Create Red bounding boxes  #starting point, ending point size of 2
                    cv2.rectangle(birdFrame, (left, top), (right, bottom), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (box[0], box[2]), (box[1], box[3]), (0, 255, 0), 2)
                    cv2.rectangle(birdFrame, (left, top), (right, bottom), (0, 255, 0), 2)


            text = "No of at-risk people: %s" % str(math.ceil(int(len(redZoneList))/3))          # Count People at Risk
            location = (10,25)                          # Set the location of the displayed text
            cv2.putText(frame, text, location, cv2.FONT_HERSHEY_SIMPLEX, 1, (246,86,86), 2, cv2.LINE_AA)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        
        finalframe=np.concatenate((birdFrame, frame), axis = 1)
        
        cv2.imshow("rgb", finalframe)

        if cv2.waitKey(1) == ord('q'):
            break
