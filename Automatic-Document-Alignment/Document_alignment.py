import cv2
import numpy as np

def order_points(pts):
	'''Rearrange coordinates to order: 
       top-left, top-right, bottom-right, bottom-left'''
	rect = np.zeros((4, 2), dtype='float32')
	pts = np.array(pts)
	s = pts.sum(axis=1)
	# Top-left point will have the smallest sum.
	rect[0] = pts[np.argmin(s)]
    # Bottom-right point will have the largest sum.
	rect[2] = pts[np.argmax(s)]
	
	diff = np.diff(pts, axis=1)
    # Top-right point will have the smallest difference.
	rect[1] = pts[np.argmin(diff)]
    # Bottom-left will have the largest difference.
	rect[3] = pts[np.argmax(diff)]
	# return the ordered coordinates
	return rect.astype('int').tolist()



def scan(img):
    if max(img.shape) > 512:
        img = cv2.resize(img, None, fx=0.5, fy=0.5)
    orig_img = img.copy()
    # Repeated Closing operation to remove text from the document.
    kernel = np.ones((5,5),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations= 3)
    # GrabCut
    mask = np.zeros(img.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (20,20,img.shape[1]-20,img.shape[0]-20)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    gray = cv2.GaussianBlur(gray, (11, 11), 0)
    # Edge Detection.
    canny = cv2.Canny(gray, 0, 200)
    canny = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    
    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # Keeping only the largest detected contour.
    page = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Detecting Edges through Contour approximation.
    # Loop over the contours.
    if len(page) == 0:
        return orig_img
    for c in page:
        # Approximate the contour.
        epsilon = 0.02 * cv2.arcLength(c, True)
        corners = cv2.approxPolyDP(c, epsilon, True)
        # If our approximated contour has four points.
        if len(corners) == 4:
            break
    # Sorting the corners and converting them to desired shape.
    corners = sorted(np.concatenate(corners).tolist())
    # For 4 corner points being detected.
    # Rearranging the order of the corner points.
    corners = order_points(corners)
    
    (tl, tr, br, bl) = corners
    # Finding the maximum width.
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Finding the maximum height.
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # Final destination co-ordinates.
    destination_corners = [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]]
    
    h, w = orig_img.shape[:2]
    # Getting the homography.
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    # Perspective transform using homography.
    un_warped = cv2.warpPerspective(orig_img, M, (w, h), flags=cv2.INTER_LINEAR)
    # Crop
    final = un_warped[:destination_corners[2][1], :destination_corners[2][0]]
    return final

if __name__ == 'main':
    img = cv2.imread('inputs/scanned-form.jpg')

    scanned = scan(img)
    cv2.imwrite('grabcutop/aligned.jpg.jpg', scanned)