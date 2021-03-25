import cv2

"""
Contour detection and drawing using different extraction modes to complement 
the understanding of hierarchies
"""
image2 = cv2.imread('../../input/custom_colors.jpg')
img_gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
ret, thresh2 = cv2.threshold(img_gray2, 150, 255, cv2.THRESH_BINARY)

contours3, hierarchy3 = cv2.findContours(thresh2, cv2.RETR_LIST, 
                                       cv2.CHAIN_APPROX_NONE)
image_copy4 = image2.copy()
cv2.drawContours(image_copy4, contours3, -1, (0, 255, 0), 2, 
                 cv2.LINE_AA)
# see the results
cv2.imshow('LIST', image_copy4)
print(f"LIST: {hierarchy3}")
cv2.waitKey(0)
cv2.imwrite('contours_retr_list.jpg', image_copy4)
cv2.destroyAllWindows()

contours4, hierarchy4 = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_NONE)
image_copy5 = image2.copy()
cv2.drawContours(image_copy5, contours4, -1, (0, 255, 0), 2, 
                 cv2.LINE_AA)
# see the results
cv2.imshow('EXTERNAL', image_copy5)
print(f"EXTERNAL: {hierarchy4}")
cv2.waitKey(0)
cv2.imwrite('contours_retr_external.jpg', image_copy5)
cv2.destroyAllWindows()

contours5, hierarchy5 = cv2.findContours(thresh2, cv2.RETR_CCOMP, 
                                       cv2.CHAIN_APPROX_NONE)
image_copy6 = image2.copy()
cv2.drawContours(image_copy6, contours5, -1, (0, 255, 0), 2, 
                 cv2.LINE_AA)
# see the results
cv2.imshow('CCOMP', image_copy6)
print(f"CCOMP: {hierarchy5}")
cv2.waitKey(0)
cv2.imwrite('contours_retr_ccomp.jpg', image_copy6)
cv2.destroyAllWindows()

contours6, hierarchy6 = cv2.findContours(thresh2, cv2.RETR_TREE, 
                                       cv2.CHAIN_APPROX_NONE)
image_copy7 = image2.copy()
cv2.drawContours(image_copy7, contours6, -1, (0, 255, 0), 2, 
                 cv2.LINE_AA)
# see the results
cv2.imshow('TREE', image_copy7)
print(f"TREE: {hierarchy6}")
cv2.waitKey(0)
cv2.imwrite('contours_retr_tree.jpg', image_copy7)
cv2.destroyAllWindows()
