import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
import math 

def houghTransform(img, theta_seg):
    height = img.shape[0]
    width = img.shape[1] 

    tan_from_origin = round(np.hypot(height, width))    # Must, use this method for the input.bmp to work
    theta_param = np.deg2rad(np.arange(0, 180, theta_seg))   # Ranges from 0 to size of image
    rho_param = np.arange(-tan_from_origin, tan_from_origin)    # Ranges from -height of image to max height of image

    total_thetas = len(theta_param)
    total_rho = tan_from_origin * 2
    accumulator = np.zeros((total_rho, total_thetas))   # Initalize to zero, array size relative to rho, theta
    # Edge Point (x, y)
    edgeY, edgeX = np.nonzero(img) 
    # Vote Accumulator
    for i in range(width):
        # Loop through edge pixel
        x = edgeX[i]
        y = edgeY[i]
        for j in range(total_thetas):
            rho = round((x * np.cos(theta_param[j])) + (y * np.sin(theta_param[j]))) + tan_from_origin
            accumulator[rho, j] = accumulator[rho, j] + 1

    return accumulator, theta_param, rho_param

# TODO: Line Detection [Bonus]
def houghLine(origImage, accumulator, thetas, rhos):
    result = np.zeros(origImage.shape)    
    return result

def showImages(origImage, cannyImg, accumulator, thetas, rhos):
    fig, ax = plt.subplots(1, 3, figsize=(15, 10))
    
    # Original Image
    ax[0].imshow(origImage)
    ax[0].set_title('Original')
    
    # Edge Detection Image
    ax[1].imshow(cannyImg, cmap=plt.cm.gray)
    ax[1].set_title('Edge Detect')

    # Hough Transform Image
    ax[2].imshow(
        accumulator, cmap=plt.cm.gist_earth, origin='lower',
        extent=[
            np.rad2deg(thetas[0]), 
            np.rad2deg(thetas[-1]), 
            rhos[0], rhos[-1]
        ]
    )
    ax[2].set_title('Hough Transform')
    ax[2].set_xlabel('Theta [Degrees]')
    ax[2].set_ylabel('Rho [Pixels]')

    fig.tight_layout(pad=3.0)
    plt.show()

if __name__ == '__main__':
    imgpath = 'test.bmp'
    img = cv2.imread(imgpath)
    origImage = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cannyImage = cv2.Canny(gray, 50, 100)

    theta_seg = 1   # For Voting
    
    accumulator, thetas, rhos = houghTransform(cannyImage, theta_seg)
    # lineImage = houghLine(origImage, accumulator, thetas, rhos)
    showImages(origImage, cannyImage, accumulator, thetas, rhos)
