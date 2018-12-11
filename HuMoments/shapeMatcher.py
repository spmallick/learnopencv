import cv2

def main():

    im1 = cv2.imread("images/K.png",cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread("images/K-rotated.png",cv2.IMREAD_GRAYSCALE)
    im3 = cv2.imread("images/A.png",cv2.IMREAD_GRAYSCALE)

    m1 = cv2.matchShapes(im1,im1,cv2.CV_CONTOURS_MATCH_I1,0)
    m2 = cv2.matchShapes(im1,im2,cv2.CV_CONTOURS_MATCH_I1,0)
    m3 = cv2.matchShapes(im1,im3,cv2.CV_CONTOURS_MATCH_I1,0)
    
    print("Shape Distances Between -------------------------")

    print("K.png and K.png : {}".format(m1))
    print("K.png and K-transformed.png : {}".format(m2))
    print("K.png and A.png : {}".format(m3))

if __name__ == "__main__":
    main()