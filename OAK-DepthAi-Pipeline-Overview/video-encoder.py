import depthai as dai

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and output
camRgb = pipeline.create(dai.node.ColorCamera)
videoEnc = pipeline.create(dai.node.VideoEncoder)
xout = pipeline.create(dai.node.XLinkOut)

xout.setStreamName('h265')

# Properties
camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_4_K)
videoEnc.setDefaultProfilePreset(3840, 2160, 30, dai.VideoEncoderProperties.Profile.H265_MAIN)

# Linking
camRgb.video.link(videoEnc.input)
videoEnc.bitstream.link(xout.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the encoded data from the output defined above
    q = device.getOutputQueue(name="h265", maxSize=30, blocking=True)

    # The .h265 file is a raw stream file (not playable yet)
    with open('video.h265', 'wb') as videoFile:
        print("Press Ctrl+C to stop encoding...")
        try:
            while True:
                h265Packet = q.get()  # Blocking call, will wait until a new data has arrived
                h265Packet.getData().tofile(videoFile)  # Appends the packet data to the opened file
        except KeyboardInterrupt:
            # Keyboard interrupt (Ctrl + C) detected
            pass

    print("To view the encoded data, convert the stream file (.h265) into a video file (.mp4) using a command below:")
    print("ffmpeg -framerate 30 -i video.h265 -c copy video.mp4")