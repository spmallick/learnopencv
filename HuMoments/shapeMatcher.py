import cv2

def main():

    im1 = cv2.imread("images/S0.png",cv2.IMREAD_GRAYSCALE)
    im2 = cv2.imread("images/K0.png",cv2.IMREAD_GRAYSCALE)
    im3 = cv2.imread("images/S4.png",cv2.IMREAD_GRAYSCALE)

    m1 = cv2.matchShapes(im1,im1,cv2.CONTOURS_MATCH_I2,0)
    m2 = cv2.matchShapes(im1,im2,cv2.CONTOURS_MATCH_I2,0)
    m3 = cv2.matchShapes(im1,im3,cv2.CONTOURS_MATCH_I2,0)

    print("Shape Distances Between \n-------------------------")

    print("S0.png and S0.png : {}".format(m1))
    print("S0.png and K0.png : {}".format(m2))
    print("S0.png and S4.png : {}".format(m3))

if __name__ == "__main__":
    main()
