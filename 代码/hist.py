import numpy as np
import matplotlib.pyplot as plt
def histeq(img, nbr_bins=256):
    """ Histogram equalization of a grayscale image. """

    imhist, bins = np.histogram(img.flatten(), nbr_bins, normed=True)


    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = 255 * cdf / cdf[-1]


    result = np.interp(img.flatten(), bins[:-1], cdf)

    return result.reshape(img.shape), cdf

def show_hist(data):
    plt.figure("hist")
    arr = data.flatten()
    n, bins, patches = plt.hist(arr, bins=256, normed=1, edgecolor='None', facecolor='red')
    plt.show()
