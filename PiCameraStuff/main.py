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
from scipy import signal
from PIL import Image

from scipy.ndimage import gaussian_filter


def isolate_color(im):
    boundary_pink = [([100, 100, 100], [200, 200, 200])]

    for (lower, upper) in boundary_pink:
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        mask = cv2.inRange(im, lower, upper)
        output = cv2.bitwise_and(im, im, mask=mask)

    # cv2.imwrite(imagename + "_edited.jpg", output)
    return output

#camera = PiCamera()
#camera.resolution = (width, height)
#camera.color_effects = (128, 128)

path = pathlib.Path().absolute()

imagename =  "red_dots_plotted"
imagenamepng = imagename + ".jpg"


#camera.start_preview()
#sleep(2)
#camera.capture(imagenamejpg)
#camera.stop_preview()

absolute_path = os.path.join(os.getcwd(), imagenamepng)
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image")
args = vars(ap.parse_args())


image = cv2.imread("Images/" + imagenamepng)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
unedited = image
block_size = 10

# antall blocks som skal lages, ut ifra bildens størrelse og block_size
i_max = int(len(image)/block_size)
j_max = int(len(image[0])/block_size)

# initialisere endelig array
color_avg = np.zeros(shape=(i_max, j_max))

image = isolate_color(image)
#plt.imshow(image, interpolation="none")

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
fig = plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(unedited, interpolation='none')
fig.add_subplot(1, 2, 2)
plt.imshow(color_avg, interpolation='none')
#plt.imshow(image, interpolation="none")
#plt.imsave(f"Images/{imagename}_color_average.png", color_avg)
plt.show()

#Butterworth
b, a = signal.butter(20, 0.1)



Rchan = image[:, :, 0]

fs = Rchan.shape[1] #Samples per meter (bredde på bildet, bare definert vilkårlig)

Rchan = Rchan.flatten()
# FFT

# 100 prikker per meter
# med fs = 1200 samples /meter vil det være 120 samples per prikk

#zeros = np.argwhere(Rchan == 0)
#zeros = zeros[:, 0]
#zeros = zeros[1::3]
#Rchan = np.delete(Rchan, zeros)
#Rchan = Rchan - np.average(Rchan)

Rchan = signal.filtfilt(b, a, Rchan)
Rchan = Rchan - np.average(Rchan)
plt.plot(Rchan)
plt.show()

freq = np.fft.fftfreq(len(Rchan))
fft = np.fft.fft(Rchan)
plt.plot(freq, fft)
plt.show()

q = abs(freq[np.nanargmax(fft)])
print(q)

# invert image
#image = (255-image)
#cv2.imwrite(imagename + "inverted.jpg", image)

# Gaussian Filter:
#image = gaussian_filter(image, sigma = 10)
#cv2.imwrite(imagename + "gaussfiltered.jpg", image)



