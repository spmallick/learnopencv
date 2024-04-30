import cv2
import numpy as np

# Callback function for drawing the rectangle
def draw_mask(event, x, y, flags, param):
    global drawing, img_mask
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(img_mask, (x, y), 10, (255), -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(img_mask, (x, y), 10, (255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img_mask, (x, y), 10, (255), -1)

# Load an image
img = cv2.imread('/path/to/your/image')
#img = cv2.resize(img, (512,512))
img_mask = np.zeros(img.shape[:2], dtype=np.uint8)

drawing = False

# Create a window and bind the function to window
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_mask)

while True:
    # Display the image
    cv2.imshow('image', cv2.addWeighted(img, 0.8, cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR), 0.2, 0))
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # Press ESC to exit
        break

# Create a binary mask
binary_mask = (img_mask > 0).astype(np.uint8) * 255

# Save the binary mask
cv2.imwrite('saved_mask.png', binary_mask)

# Show the binary mask and wait until any key is pressed
cv2.imshow('Binary Mask', binary_mask)
cv2.waitKey(0)

cv2.destroyAllWindows()