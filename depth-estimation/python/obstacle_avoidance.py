import numpy as np 
import cv2


# Check for left and right camera IDs
# These values can change depending on the system
CamL_id = 2 # Camera ID for left camera
CamR_id = 0 # Camera ID for right camera


CamL= cv2.VideoCapture(CamL_id)
CamR= cv2.VideoCapture(CamR_id)

# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("../data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()

disparity = None
depth_map = None

# These parameters can vary according to the setup
max_depth = 400 # maximum distance the setup can measure (in cm)
min_depth = 50 # minimum distance the setup can measure (in cm)
depth_thresh = 100.0 # Threshold for SAFE distance (in cm)

# Reading the stored the StereoBM parameters
cv_file = cv2.FileStorage("../data/depth_estmation_params_py.xml", cv2.FILE_STORAGE_READ)
numDisparities = int(cv_file.getNode("numDisparities").real())
blockSize = int(cv_file.getNode("blockSize").real())
preFilterType = int(cv_file.getNode("preFilterType").real())
preFilterSize = int(cv_file.getNode("preFilterSize").real())
preFilterCap = int(cv_file.getNode("preFilterCap").real())
textureThreshold = int(cv_file.getNode("textureThreshold").real())
uniquenessRatio = int(cv_file.getNode("uniquenessRatio").real())
speckleRange = int(cv_file.getNode("speckleRange").real())
speckleWindowSize = int(cv_file.getNode("speckleWindowSize").real())
disp12MaxDiff = int(cv_file.getNode("disp12MaxDiff").real())
minDisparity = int(cv_file.getNode("minDisparity").real())
M = cv_file.getNode("M").real()
cv_file.release()

# mouse callback function
def mouse_click(event,x,y,flags,param):
	global Z
	if event == cv2.EVENT_LBUTTONDBLCLK:
		print("Distance = %.2f cm"%depth_map[y,x])	


cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
cv2.setMouseCallback('disp',mouse_click)

output_canvas = None

# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()

def obstacle_avoid():

	# Mask to segment regions with depth less than threshold
	mask = cv2.inRange(depth_map,10,depth_thresh)

	# Check if a significantly large obstacle is present and filter out smaller noisy regions
	if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:

		# Contour detection 
		contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnts = sorted(contours, key=cv2.contourArea, reverse=True)
		
		# Check if detected contour is significantly large (to avoid multiple tiny regions)
		if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:

			x,y,w,h = cv2.boundingRect(cnts[0])

			# finding average depth of region represented by the largest contour 
			mask2 = np.zeros_like(mask)
			cv2.drawContours(mask2, cnts, 0, (255), -1)

			# Calculating the average depth of the object closer than the safe distance
			depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
			
			# Display warning text
			cv2.putText(output_canvas, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
			cv2.putText(output_canvas, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
			cv2.putText(output_canvas, "%.2f cm"%depth_mean, (x+5,y+40), 1, 2, (100,10,25), 2, 2)

	else:
		cv2.putText(output_canvas, "SAFE!", (100,100),1,3,(0,255,0),2,3)

	cv2.imshow('output_canvas',output_canvas)
	

while True:
	retR, imgR= CamR.read()
	retL, imgL= CamL.read()
	
	if retL and retR:
		
		output_canvas = imgL.copy()

		imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
		imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)

		# Applying stereo image rectification on the left image
		Left_nice= cv2.remap(imgL_gray,
							Left_Stereo_Map_x,
							Left_Stereo_Map_y,
							cv2.INTER_LANCZOS4,
							cv2.BORDER_CONSTANT,
							0)
		
		# Applying stereo image rectification on the right image
		Right_nice= cv2.remap(imgR_gray,
							Right_Stereo_Map_x,
							Right_Stereo_Map_y,
							cv2.INTER_LANCZOS4,
							cv2.BORDER_CONSTANT,
							0)

		# Setting the updated parameters before computing disparity map
		stereo.setNumDisparities(numDisparities)
		stereo.setBlockSize(blockSize)
		stereo.setPreFilterType(preFilterType)
		stereo.setPreFilterSize(preFilterSize)
		stereo.setPreFilterCap(preFilterCap)
		stereo.setTextureThreshold(textureThreshold)
		stereo.setUniquenessRatio(uniquenessRatio)
		stereo.setSpeckleRange(speckleRange)
		stereo.setSpeckleWindowSize(speckleWindowSize)
		stereo.setDisp12MaxDiff(disp12MaxDiff)
		stereo.setMinDisparity(minDisparity)

		# Calculating disparity using the StereoBM algorithm
		disparity = stereo.compute(Left_nice,Right_nice)
		# NOTE: compute returns a 16bit signed single channel image,
		# CV_16S containing a disparity map scaled by 16. Hence it 
		# is essential to convert it to CV_16S and scale it down 16 times.

		# Converting to float32 
		disparity = disparity.astype(np.float32)

		# Normalizing the disparity map
		disparity = (disparity/16.0 - minDisparity)/numDisparities
		
		depth_map = M/(disparity) # for depth in (cm)

		mask_temp = cv2.inRange(depth_map,min_depth,max_depth)
		depth_map = cv2.bitwise_and(depth_map,depth_map,mask=mask_temp)

		obstacle_avoid()
		
		cv2.resizeWindow("disp",700,700)
		cv2.imshow("disp",disparity)

		if cv2.waitKey(1) == 27:
			break
	
	else:
		CamL= cv2.VideoCapture(CamL_id)
		CamR= cv2.VideoCapture(CamR_id)