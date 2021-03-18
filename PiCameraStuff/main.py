# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#from picamera import PiCamera
from time import sleep
import numpy as np
import cv2
import argparse
import os
import pathlib
import matplotlib.pyplot as plt
from PIL import Image

from scipy.ndimage import gaussian_filter


#camera = PiCamera()
#camera.resolution = (width, height)
#camera.color_effects = (128, 128)

path = pathlib.Path().absolute()

imagename = input("bildenavn: ")
imagenamejpg = imagename ".jpg"

#camera.start_preview()
#sleep(2)
#camera.capture(imagenamejpg)
#camera.stop_preview()

absolute_path = os.path.join(os.getcwd(), imagenamejpg)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help = "path to the image")
args = vars(ap.parse_args())
m, n = 16, 10

image = cv2.imread("Images/test_png.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

block_size = 100

# antall blocks som skal lages, ut ifra bildens størrelse og block_size
i_max = int(len(image)/block_size)
j_max = int(len(image[0])/block_size)

# initialisere endelig array
color_avg = np.zeros(shape=(i_max, j_max))

# Deler opp bilde i størrelse spesifisert med block_size og looper over disse
for i in range(i_max):
    for j in range(j_max):

        # loope over hver pixel inni hver av disse blockene og kalkulere gjennomsnitlig av fargen rød (index 0)
        color_calc = 0
        for x in range(block_size):
            for y in range(block_size):
                color_calc += image[i*block_size + x][j*block_size + y][0] # siste index er for fargen rød
        color_calc = color_calc/(block_size**2)
        color_avg[i][j] = color_calc

# vise bilde
plt.imshow(color_avg, interpolation='none')
plt.imshow(image, interpolation="none")
plt.show()



# invert image
#image = (255-image)
#cv2.imwrite(imagename + "inverted.jpg", image)

# Gaussian Filter:
#image = gaussian_filter(image, sigma = 10)
#cv2.imwrite(imagename + "gaussfiltered.jpg", image)


#boundary_pink = [([100, 100, 100], [200, 200, 200])]

#for (lower, upper) in boundary_pink:
#    lower = np.array(lower, dtype = "uint8")
#    upper = np.array(upper, dtype = "uint8")

 #   mask = cv2.inRange(image, lower, upper)
#    output = cv2.bitwise_and(image, image, mask = mask)

  #  cv2.imwrite(imagename + "_edited.jpg", output)
