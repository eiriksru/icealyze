

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc
import cv2
from shapely.geometry import Point
from shapely.ops import cascaded_union

imagename = "scatter_plot2"
imagenamepng = imagename + ".jpg"

n = 100
size = 0.02
alpha = 0.5
Rchan = np.zeros(500)

def points():
    x = np.linspace(0, 10, 1)
    y = np.linspace(0, 10, 1)
    return x, y

def plot_scatter():

    x1, y1 = points()
    fig = plt.figure(figsize=(10, 10), frameon=False)
    ax = fig.add_subplot(111)
    alpha = 0.5
    ax.scatter(x1, y1, s=100, lw=0, color=[1., alpha, alpha])
    #ax.scatter(x2, y2, s=100, lw=0, color=[alpha, alpha, 1.])
    ax.set_axis_off()
    fig.add_axes(ax)


plot_scatter()
#plt.savefig("scatter_plot2", facecolor="w")
# Want to print all red cells along rows


plt.show()
# Read picture to an array with shape (1600, 2600, 3)
image = cv2.imread(imagenamepng)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = np.array(image)

Rchan = image[:, 1, 0]
plt.plot(Rchan)
plt.show()

