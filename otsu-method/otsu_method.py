import cv2
from matplotlib.ticker import FuncFormatter
from matplotlib import pyplot as plt
from otsu_implementation import otsu_implementation


def call_otsu_threshold(img_title="boat.jpg", is_reduce_noise=False):
    # Read the image in a greyscale mode
    image = cv2.imread(img_title, 0)

    # Apply GaussianBlur to reduce image noise if it is required
    if is_reduce_noise:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # View initial image histogram
    plt.hist(image.ravel(), 256)
    plt.xlabel('Colour intensity')
    plt.ylabel('Number of pixels')
    plt.savefig("image_hist.png")
    plt.close()

    # Applying Otsu's method setting the flag value into cv.THRESH_OTSU.
    # Use bimodal image as an input.
    # Optimal threshold value is determined automatically.
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    print("Obtained threshold: ", otsu_threshold)

    # View the resulting image histogram
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(image_result.ravel(), 256)
    ax.set_xlabel('Colour intensity')
    ax.set_ylabel('Number of pixels')
    # Get rid of 1e7
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: ('%1.1fM') % (x*1e-6)))
    plt.savefig("image_hist_result.png")
    plt.close()

    # Visualize the image after the Otsu's method application
    cv2.imshow("Otsu's thresholding result", image_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    call_otsu_threshold()
    otsu_implementation()


if __name__ == "__main__":
    main()
