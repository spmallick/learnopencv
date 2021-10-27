import cv2
import numpy as np
from mss import mss
from tkinter import *
import tkinter.messagebox
import pyautogui as gui


def getMatches(ref_trex, captured_screen):
    # Initialize lists
    list_kp2 = []
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(ref_trex, None)
    kp2, des2 = orb.detectAndCompute(captured_screen, None)
    # Match features.
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(des1, des2, None)
    # Convert to list.
    matches = list(matches)
    # Sort matches by score.
    matches.sort(key=lambda x: x.distance, reverse=False)
    # Retain only the top 25% of matches.
    numGoodMatches = int(len(matches) * 0.25)
    matches = matches[:numGoodMatches]
    # Visualize matches.
    match_img = cv2.drawMatches(ref_trex, kp1, captured_screen, kp2, matches[:50], None)
    # For each match...
    for mat in matches:
        # Get the matching keypoints for each of the images.
        img2_idx = mat.trainIdx
        # Get the coordinates.
        (x2, y2) = kp2[img2_idx].pt
        # Append to each list.
        list_kp2.append((int(x2), int(y2)))
    cv2.imshow('Matches', cv2.resize(match_img, None, fx=0.5, fy=0.5))
    cv2.imwrite('Matches.jpg', match_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return list_kp2


def drawBboxManual(action, x, y, flags, *userdata):
    global bbox_top_left, bbox_bottom_right
    org_x = int(6 * img.shape[1] / 12)
    org_y =  int(3 * img.shape[0] / 5)
    cv2.putText(img, 'Error detecting Trex', (org_x + 20, org_y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'Please click and drag', (org_x + 20, org_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(img, 'To define the target area', (org_x + 20, org_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 1, cv2.LINE_AA)
    if action == cv2.EVENT_LBUTTONDOWN:
        bbox_top_left = [(x, y)]
        cv2.circle(img, (bbox_top_left[0][0], bbox_top_left[0][1]), 3, (0, 0, 255), -1)
        cv2.imshow("DetectionArea", img)
    if action == cv2.EVENT_LBUTTONUP:
        bbox_bottom_right = [(x, y)]
        cv2.circle(img, (bbox_bottom_right[0][0], bbox_bottom_right[0][1]), 3, (0, 0, 255), -1)
        cv2.rectangle(img, (bbox_top_left[0][0], bbox_top_left[0][1]),
                      (bbox_bottom_right[0][0], bbox_bottom_right[0][1]), (0, 255, 0), 2)
        cv2.imshow("DetectionArea", img)
    cv2.imshow("DetectionArea", img)
    # cv2.imwrite('MouseDefinedBox.jpg', img)


def checkDayOrNight(img):
    pixels_intensities = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h = int(img.shape[0] / 4)
    w = int(img.shape[1] / 4)
    for i in range(h):
        for j in range(w):
            pixels_intensities.append(img[i, j])
    val = int(sum(pixels_intensities) / len(pixels_intensities))
    if val > 195:
        return True
    else:
        return False

#===============================INITILAIZATIONS=============================#

# Set keypress delay to 0.
gui.PAUSE = 0

# Initialize lists to hold bounding box coordinates.
bbox_top_left = []
bbox_bottom_right = []

#============================================================================#


# Main function.
if __name__ == "__main__":
    # Load reference image.
    ref_img = cv2.imread('trex.png')
    # Uncomment the following line if you are on Dark Mode.
    # ref_img = cv2.imread('tRexDark.jpg')
    screen = mss()
    # Identify the display to capture.
    monitor = screen.monitors[1]
    # Check resolution info returned by mss.
    print('MSS resolution info : ', monitor)
    # Grab the screen.
    screenshot = screen.grab(monitor)
    # Convert to numpy array.
    screen_img = np.array(screenshot)
    # Define Tested height and width of TRex, according to the screen resolution.
    hTrex = int((0.062962) * screen_img.shape[0])
    wTrex = int((0.046875) * screen_img.shape[1])
    tested_area = hTrex * wTrex
    print('Tested Dimensions : ', hTrex, '::', wTrex)
    # Obtain keypoints.
    trex_keypoints = getMatches(ref_img, screen_img)
    # Convert to numpy array.
    kp_arary = np.array(trex_keypoints)
    # Get dimensions of the bounding rectangle.
    x, y, w, h = cv2.boundingRect(np.int32(kp_arary))
    obtained_area = w * h
    # print('Obtained Area : ', obtained_area)
    # tested_area = wTrex * hTrex
    # print('Tested Area : ', tested_area)

    """
    Check whether matches are good by comparing the area of the boundingRect to 
    the tested area. If the obtained bounding box is not too small or too large.
    """
    if 0.1*tested_area < obtained_area < 3*tested_area:
        print('Matches are good.')
        # Set Target area bbox coordinates.
        xRoi1 = x + wTrex
        yRoi1 = y
        xRoi2 = x + 2 * wTrex
        """
        Set the height of bbox at 50%  of original. To make sure that it does not 
        capture the line below the T-Rex. You can play with this value to come up
        with better positioning.
        """
        yRoi2 = y + int(0.5*hTrex)
        # Draw rectangle.
        cv2.rectangle(screen_img, (xRoi1, yRoi1), (xRoi2, yRoi2), (0, 255, 0), 2)
        cv2.imshow('DetectionArea', cv2.resize(screen_img, None, fx=0.5, fy=0.5))
        cv2.imwrite('ScreenBox.jpg', screen_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print('Matches are not good, please set target area manually.')
        img = cv2.resize(screen_img, None, fx=0.5, fy=0.5)
        cv2.namedWindow('DetectionArea')
        cv2.setMouseCallback('DetectionArea', drawBboxManual)
        cv2.imshow('DetectionArea', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Resize back and set Target area bbox coordinates accordingly.
        xRoi1 = 2 * bbox_top_left[0][0]
        yRoi1 = 2 * bbox_top_left[0][1]
        xRoi2 = 2 * bbox_bottom_right[0][0]
        yRoi2 = 2 * bbox_bottom_right[0][1]

    # If you click-drag performed wrong, restart.
    if xRoi1 == xRoi2 and yRoi1 == yRoi2:
        print('Please draw the bounding box again using click-drag-release method not click-drag-click')
        window = Tk()
        window.wm_withdraw()
        # Error message at the center of the screen.
        window.geometry("1x1+"+str(window.winfo_screenwidth()//2)+"+"+str(window.winfo_screenheight()//2))
        tkinter.messagebox.showinfo(title="Error", message="Please use click-drag-release")
        exit()

  
    """
    If the  screen resolution returned by mss is different to that of actual system  resolution.
    That  could mean multiple connected high resolution displays or displays with  auto scaling
    feature,  such as Macbooks with retina display. We found no issue upto  1920 x 1080 windows 
    and Linux  systems. Macbook air without retina display (1366 x 768) works without any issue
    as well. However, Macbooks with retina display (2560 x 1600) has some issue with mss screen
    capture.  The tested scaling factor is half for  2560 x 1600  Macs. Uncomment the following 
    line in that case. You may need to check the scaling factor for other cases. 

    """

    # xRoi1, yRoi1, xRoi2, yRoi2 = (xRoi1 // 2, yRoi1 // 2, xRoi2 // 2, yRoi2 // 2)

    # Create a dictionary for MSS, defining size of the screen to be captured.
    obstacle_check_bbox = {'top': yRoi1, 'left': xRoi1, 'width': xRoi2 - xRoi1, 'height': yRoi2 - yRoi1}
    # Day or Night mode checking patch. Estimated just above obstacle detecting patch.
    day_check_bbox    = {'top': yRoi1 - 2*hTrex, 'left': xRoi1, 'width': xRoi2, 'height': yRoi2 - 2*hTrex}

    # Main loop.
    while True:
        # Capture obstacle detecting patch.
        obstacle_check_patch = screen.grab(obstacle_check_bbox)
        obstacle_check_patch = np.array(obstacle_check_patch)

        # Day or Night mode checking patch.
        day_check_patch = screen.grab(day_check_bbox)
        day_check_patch = np.array(day_check_patch)

        # Convert obstacle detecting area to gray scale.
        obstacle_check_gray = cv2.cvtColor(obstacle_check_patch, cv2.COLOR_BGR2GRAY)

        # Check the game mode.
        day = checkDayOrNight(day_check_patch)

        # Perform contour analysis according to the game mode.
        if day:
            # Add 10px padding for effective contour analysis.
            obstacle_check_gray = cv2.copyMakeBorder(obstacle_check_gray, 10, 10, 10, 10,
                                             cv2.BORDER_CONSTANT, None, value=255)
            # Perform thresholding.
            ret, thresh = cv2.threshold(obstacle_check_gray, 127, 255,
                                        cv2.THRESH_BINARY)
        else:
            # Add 10px padding for effective contour analysis.
            obstacle_check_gray = cv2.copyMakeBorder(obstacle_check_gray, 10, 10, 10, 10,
                                             cv2.BORDER_CONSTANT, None, value=0)
            # Perform thresholding.
            ret, thresh = cv2.threshold(obstacle_check_gray, 127, 255,
                                        cv2.THRESH_BINARY_INV)

        # Find contours.
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_NONE)
        # Print number of contours.
        # print('Contours Detected : ', len(contours))

        # Make T-Rex jump.
        if len(contours) > 1:
            gui.press('space', interval=0.1)

        cv2.imshow('Window', obstacle_check_gray)
        key = cv2.waitKey(1)
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
