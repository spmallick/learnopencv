# Import numpy and OpenCV
import numpy as np
import cv2


def movingAverage(curve, radius): 
  window_size = 2 * radius + 1
  # Define the filter 
  f = np.ones(window_size)/window_size 
  # Add padding to the boundaries 
  curve_pad = np.lib.pad(curve, (radius, radius), 'edge') 
  # Apply convolution 
  curve_smoothed = np.convolve(curve_pad, f, mode='same') 
  # Remove padding 
  curve_smoothed = curve_smoothed[radius:-radius]
  # return smoothed curve
  return curve_smoothed 

def smooth(trajectory): 
  smoothed_trajectory = np.copy(trajectory) 
  # Filter the x, y and angle curves
  for i in range(3):
    smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)

  return smoothed_trajectory

def fixBorder(frame):
  s = frame.shape
  # Scale the image 4% without moving the center
  T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.04)
  frame = cv2.warpAffine(frame, T, (s[1], s[0]))
  return frame


# The larger the more stable the video, but less reactive to sudden panning
SMOOTHING_RADIUS=50 

# Read input video
cap = cv2.VideoCapture('video.mp4') 
 
# Get frame count
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
 
# Get width and height of video stream
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Get frames per second (fps)
fps = cap.get(cv2.CAP_PROP_FPS)
 
# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
 
# Set up output video
out = cv2.VideoWriter('video_out.avi', fourcc, fps, (2 * w, h))

# Read first frame
_, prev = cap.read() 
 
# Convert frame to grayscale
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY) 

# Pre-define transformation-store array
transforms = np.zeros((n_frames-1, 3), np.float32) 

for i in range(n_frames-2):
  # Detect feature points in previous frame
  prev_pts = cv2.goodFeaturesToTrack(prev_gray,
                                     maxCorners=200,
                                     qualityLevel=0.01,
                                     minDistance=30,
                                     blockSize=3)
   
  # Read next frame
  success, curr = cap.read() 
  if not success: 
    break 

  # Convert to grayscale
  curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY) 

  # Calculate optical flow (i.e. track feature points)
  curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None) 

  # Sanity check
  assert prev_pts.shape == curr_pts.shape 

  # Filter only valid points
  idx = np.where(status==1)[0]
  prev_pts = prev_pts[idx]
  curr_pts = curr_pts[idx]

  #Find transformation matrix
  m = cv2.estimateRigidTransform(prev_pts, curr_pts, fullAffine=False) #will only work with OpenCV-3 or less
   
  # Extract traslation
  dx = m[0,2]
  dy = m[1,2]

  # Extract rotation angle
  da = np.arctan2(m[1,0], m[0,0])
   
  # Store transformation
  transforms[i] = [dx,dy,da]
   
  # Move to next frame
  prev_gray = curr_gray

  print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))

# Compute trajectory using cumulative sum of transformations
trajectory = np.cumsum(transforms, axis=0) 
 
# Create variable to store smoothed trajectory
smoothed_trajectory = smooth(trajectory) 

# Calculate difference in smoothed_trajectory and trajectory
difference = smoothed_trajectory - trajectory
 
# Calculate newer transformation array
transforms_smooth = transforms + difference

# Reset stream to first frame 
cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
 
# Write n_frames-1 transformed frames
for i in range(n_frames-2):
  # Read next frame
  success, frame = cap.read() 
  if not success:
    break

  # Extract transformations from the new transformation array
  dx = transforms_smooth[i,0]
  dy = transforms_smooth[i,1]
  da = transforms_smooth[i,2]

  # Reconstruct transformation matrix accordingly to new values
  m = np.zeros((2,3), np.float32)
  m[0,0] = np.cos(da)
  m[0,1] = -np.sin(da)
  m[1,0] = np.sin(da)
  m[1,1] = np.cos(da)
  m[0,2] = dx
  m[1,2] = dy

  # Apply affine wrapping to the given frame
  frame_stabilized = cv2.warpAffine(frame, m, (w,h))

  # Fix border artifacts
  frame_stabilized = fixBorder(frame_stabilized) 

  # Write the frame to the file
  frame_out = cv2.hconcat([frame, frame_stabilized])

  # If the image is too big, resize it.
  if(frame_out.shape[1] > 1920): 
    frame_out = cv2.resize(frame_out, (frame_out.shape[1]/2, frame_out.shape[0]/2));
  
  cv2.imshow("Before and After", frame_out)
  cv2.waitKey(10)
  out.write(frame_out)

# Release video
cap.release()
out.release()
# Close windows
cv2.destroyAllWindows()