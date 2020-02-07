import matplotlib.pyplot as plt
from PIL import Image
import skimage
from skimage import io,measure
import cv2
import numpy as np
import scipy.sparse.linalg as sla
import scipy.misc
import os



def find_entropy(MyImg):
    hist = np.histogramdd(np.ravel(MyImg), bins=256)           #### hist values of input image
    # print('Image size:', MyImg.size)
    prob = hist[0]/(MyImg.size)                                 #### get probability values
    # prob = hist[0]/(MyImg.size[0]*MyImg.size[1])
    prob = list(filter(lambda p: p > 0, np.ravel(prob)))        #### filter zero prob
    entropy_s = -np.sum(np.multiply(prob, np.log2(prob)))       #### calc entropy
    return entropy_s


if __name__=='__main__':

    imgdir = os.path.dirname(os.path.realpath(__file__))
    print(imgdir)

    file = 'img1.png'      ## change 'img1.png to lena.jpg to check entropy

    # MyImg = cv2.imread(os.path.join(imgdir, file), 0)      # image 1

    MyImg = cv2.imread(os.path.join(imgdir, 'lena.jpg'), 0)    # image 2


    # plt.hist(MyImg.ravel(), 256, [0, 256])
    # plt.title('Original histogram')
    # plt.xlabel('Pixel intensity')
    # plt.ylabel('Frequency of occurrence')
    # plt.show()

    # histogram equalization

    equ = cv2.equalizeHist(MyImg)
    cv2.imwrite(os.path.join(imgdir, 'HE1.png'), equ)

    # plt.hist(equ.ravel(), 256, [0, 256])
    # plt.title('Equalised histogram')
    # plt.xlabel('Pixel intensity')
    # plt.ylabel('Frequency of occurrence')
    # plt.show()
    print('Entropy:', find_entropy(MyImg))
    print('Reduced Entropy:', find_entropy(equ))

    cv2.namedWindow('original',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('original', 600, 600)

    cv2.namedWindow('Histogram equalize', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Histogram equalize', 600, 600)

    cv2.imshow('original', MyImg)
    cv2.imshow("Histogram equalize", equ)

    cv2.waitKey()
    cv2.destroyAllWindows()

''' ######### ANOTHER METHOD TO FIND ENTROPY ##############
entropy = skimage.measure.shannon_entropy(MyImg)
entropy_r = skimage.measure.shannon_entropy(equ)
'''









