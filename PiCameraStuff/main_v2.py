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
import matplotlib.cm as cm
from PIL import Image

os.chdir(os.path.dirname(os.path.abspath(__file__)))

width = 2500
height = 1600

#camera = PiCamera(framerate=40)
#camera.resolution = (width, height)

#Gjør bildet svart hvitt
#camera.color_effects = (128, 128)

# camera setup
#camera.iso = 100

# awb_gains setup
#camera.exposure_mode = 'off'
#camera.awb_mode = 'off'
#g = camera.awb_gains
#camera.awb_gains = g


path = pathlib.Path().absolute()

#imagename = input("bildenavn: ")
#imagenamepng = imagename + ".png"

#camera.start_preview()
#sleep(2)
#camera.capture(imagenamepng)
#camera.stop_preview()

# Metoder for å smoothe ut bilde
interpolasjon_metoder = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
           'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
           'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']


# Deler opp bilde i størrelse spesifisert med block_size og looper over disse
def img_avg_block(name):
    # hente inn bilde og gjøre det til format RGB
    im = cv2.imread(name)
    print(f"shape of image = {im.shape}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    block_size = 70

    # antall blocks som skal lages, ut ifra bildens størrelse og block_size
    i_max = int(len(im)/block_size)
    j_max = int(len(im[0])/block_size)

    # initialisere endelig array
    color_avg = np.zeros(shape=(i_max, j_max))

    # Deler opp bilde i størrelse spesifisert med block_size og looper over disse
    for i in range(i_max):
        for j in range(j_max):
            # loope over hver pixel inni hver av disse blockene og kalkulere gjennomsnitlig av fargen rød (index 0)
            color_calc = 0
            for x in range(block_size):
                for y in range(block_size):
                    color_calc += im[i*block_size + x][j*block_size + y][0] # siste index er for fargen rød
            color_calc = color_calc/(block_size**2)
            color_avg[i][j] = color_calc
    return color_avg


#top_quality_ice = img_avg_block()

slitt = img_avg_block('Images/hallbesøk_22_april/IMG_2298_cropped.png')
uslitt = img_avg_block('Images/hall_uslitt.png')


diff_img = np.zeros(shape=(len(uslitt), len(uslitt[0])))


for i in range(len(slitt)):
    for j in range(len(slitt[0])):
        diff_img[i][j] = slitt[i][j] - uslitt[i][j]

plt.imshow(diff_img, interpolation=interpolasjon_metoder[16], cmap=cm.jet_r) 	# sinc blur metode
plt.show()
# invert image
#image = (255-image)
#cv2.imwrite(imagename + "inverted.jpg", image)


#for (lower, upper) in boundary_pink:
#    lower = np.array(lower, dtype = "uint8")
#    upper = np.array(upper, dtype = "uint8")

 #   mask = cv2.inRange(image, lower, upper)
#    output = cv2.bitwise_and(image, image, mask = mask)

  #  cv2.imwrite(imagename + "_edited.jpg", output)
