import cv2,time,argparse,glob
import numpy as np

#global variable to keep track of 
show = False

def onTrackbarActivity(x):
    global show
    show = True
    pass


if __name__ == '__main__' :

    # Get the filename from the command line 
    files = glob.glob('images/rub*.jpg')
    files.sort()
    # load the image 
    original = cv2.imread(files[0])
    #Resize the image
    rsize = 250
    original = cv2.resize(original,(rsize,rsize))

    #position on the screen where the windows start
    initialX = 50
    initialY = 50

    #creating windows to display images
    cv2.namedWindow('P-> Previous, N-> Next',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('SelectBGR',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('SelectHSV',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('SelectYCB',cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('SelectLAB',cv2.WINDOW_AUTOSIZE)

    #moving the windows to stack them horizontally
    cv2.moveWindow('P-> Previous, N-> Next',initialX,initialY)
    cv2.moveWindow('SelectBGR',initialX + (rsize + 5),initialY)
    cv2.moveWindow('SelectHSV',initialX + 2*(rsize + 5),initialY)
    cv2.moveWindow('SelectYCB',initialX + 3*(rsize + 5),initialY)
    cv2.moveWindow('SelectLAB',initialX + 4*(rsize + 5),initialY)

    #creating trackbars to get values for YCrCb
    cv2.createTrackbar('CrMin','SelectYCB',0,255,onTrackbarActivity)
    cv2.createTrackbar('CrMax','SelectYCB',0,255,onTrackbarActivity)
    cv2.createTrackbar('CbMin','SelectYCB',0,255,onTrackbarActivity)
    cv2.createTrackbar('CbMax','SelectYCB',0,255,onTrackbarActivity)
    cv2.createTrackbar('YMin','SelectYCB',0,255,onTrackbarActivity)
    cv2.createTrackbar('YMax','SelectYCB',0,255,onTrackbarActivity)

    #creating trackbars to get values for HSV
    cv2.createTrackbar('HMin','SelectHSV',0,180,onTrackbarActivity)
    cv2.createTrackbar('HMax','SelectHSV',0,180,onTrackbarActivity)
    cv2.createTrackbar('SMin','SelectHSV',0,255,onTrackbarActivity)
    cv2.createTrackbar('SMax','SelectHSV',0,255,onTrackbarActivity)
    cv2.createTrackbar('VMin','SelectHSV',0,255,onTrackbarActivity)
    cv2.createTrackbar('VMax','SelectHSV',0,255,onTrackbarActivity)

    #creating trackbars to get values for BGR
    cv2.createTrackbar('BGRBMin','SelectBGR',0,255,onTrackbarActivity)
    cv2.createTrackbar('BGRBMax','SelectBGR',0,255,onTrackbarActivity)
    cv2.createTrackbar('BGRGMin','SelectBGR',0,255,onTrackbarActivity)
    cv2.createTrackbar('BGRGMax','SelectBGR',0,255,onTrackbarActivity)
    cv2.createTrackbar('BGRRMin','SelectBGR',0,255,onTrackbarActivity)
    cv2.createTrackbar('BGRRMax','SelectBGR',0,255,onTrackbarActivity)

    #creating trackbars to get values for LAB
    cv2.createTrackbar('LABLMin','SelectLAB',0,255,onTrackbarActivity)
    cv2.createTrackbar('LABLMax','SelectLAB',0,255,onTrackbarActivity)
    cv2.createTrackbar('LABAMin','SelectLAB',0,255,onTrackbarActivity)
    cv2.createTrackbar('LABAMax','SelectLAB',0,255,onTrackbarActivity)
    cv2.createTrackbar('LABBMin','SelectLAB',0,255,onTrackbarActivity)
    cv2.createTrackbar('LABBMax','SelectLAB',0,255,onTrackbarActivity)

    # show all images initially
    cv2.imshow('SelectHSV',original)
    cv2.imshow('SelectYCB',original)
    cv2.imshow('SelectLAB',original)
    cv2.imshow('SelectBGR',original)
    i = 0
    while(1):

        cv2.imshow('P-> Previous, N-> Next',original)  
        k = cv2.waitKey(1) & 0xFF

        # check next image in folder    
        if k == ord('n'):
            i += 1
            original = cv2.imread(files[i%len(files)])
            original = cv2.resize(original,(rsize,rsize))
            show = True
 
        # check previous image in folder    
        elif k == ord('p'):
            i -= 1
            original = cv2.imread(files[i%len(files)])
            original = cv2.resize(original,(rsize,rsize))
            show = True
        # Close all windows when 'esc' key is pressed
        elif k == 27:
            break
        
        if show: # If there is any event on the trackbar
            show = False

            # Get values from the BGR trackbar
            BMin = cv2.getTrackbarPos('BGRBMin','SelectBGR')
            GMin = cv2.getTrackbarPos('BGRGMin','SelectBGR')
            RMin = cv2.getTrackbarPos('BGRRMin','SelectBGR')
            BMax = cv2.getTrackbarPos('BGRBMax','SelectBGR')
            GMax = cv2.getTrackbarPos('BGRGMax','SelectBGR')
            RMax = cv2.getTrackbarPos('BGRRMax','SelectBGR')
            minBGR = np.array([BMin, GMin, RMin])
            maxBGR = np.array([BMax, GMax, RMax])

            # Get values from the HSV trackbar
            HMin = cv2.getTrackbarPos('HMin','SelectHSV')
            SMin = cv2.getTrackbarPos('SMin','SelectHSV')
            VMin = cv2.getTrackbarPos('VMin','SelectHSV')
            HMax = cv2.getTrackbarPos('HMax','SelectHSV')
            SMax = cv2.getTrackbarPos('SMax','SelectHSV')
            VMax = cv2.getTrackbarPos('VMax','SelectHSV')
            minHSV = np.array([HMin, SMin, VMin])
            maxHSV = np.array([HMax, SMax, VMax])

            # Get values from the LAB trackbar
            LMin = cv2.getTrackbarPos('LABLMin','SelectLAB')
            AMin = cv2.getTrackbarPos('LABAMin','SelectLAB')
            BMin = cv2.getTrackbarPos('LABBMin','SelectLAB')
            LMax = cv2.getTrackbarPos('LABLMax','SelectLAB')
            AMax = cv2.getTrackbarPos('LABAMax','SelectLAB')
            BMax = cv2.getTrackbarPos('LABBMax','SelectLAB')
            minLAB = np.array([LMin, AMin, BMin])
            maxLAB = np.array([LMax, AMax, BMax])

            # Get values from the YCrCb trackbar
            YMin = cv2.getTrackbarPos('YMin','SelectYCB')
            CrMin = cv2.getTrackbarPos('CrMin','SelectYCB')
            CbMin = cv2.getTrackbarPos('CbMin','SelectYCB')
            YMax = cv2.getTrackbarPos('YMax','SelectYCB')
            CrMax = cv2.getTrackbarPos('CrMax','SelectYCB')
            CbMax = cv2.getTrackbarPos('CbMax','SelectYCB')
            minYCB = np.array([YMin, CrMin, CbMin])
            maxYCB = np.array([YMax, CrMax, CbMax])
            
            # Convert the BGR image to other color spaces
            imageBGR = np.copy(original)
            imageHSV = cv2.cvtColor(original,cv2.COLOR_BGR2HSV)
            imageYCB = cv2.cvtColor(original,cv2.COLOR_BGR2YCrCb)
            imageLAB = cv2.cvtColor(original,cv2.COLOR_BGR2LAB)

            # Create the mask using the min and max values obtained from trackbar and apply bitwise and operation to get the results         
            maskBGR = cv2.inRange(imageBGR,minBGR,maxBGR)
            resultBGR = cv2.bitwise_and(original, original, mask = maskBGR)         
            
            maskHSV = cv2.inRange(imageHSV,minHSV,maxHSV)
            resultHSV = cv2.bitwise_and(original, original, mask = maskHSV)
            
            maskYCB = cv2.inRange(imageYCB,minYCB,maxYCB)
            resultYCB = cv2.bitwise_and(original, original, mask = maskYCB)         
        
            maskLAB = cv2.inRange(imageLAB,minLAB,maxLAB)
            resultLAB = cv2.bitwise_and(original, original, mask = maskLAB)         
            
            # Show the results
            cv2.imshow('SelectBGR',resultBGR)
            cv2.imshow('SelectYCB',resultYCB)
            cv2.imshow('SelectLAB',resultLAB)
            cv2.imshow('SelectHSV',resultHSV)

    cv2.destroyAllWindows()

