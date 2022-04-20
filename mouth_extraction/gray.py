import os, fnmatch, sys, errno
from skimage import io
import cv2
import numpy as np
imgfile='./00004/mouth_004.png'
# img = cv2.imread(imgfile)
# # print(img.shape)
# # print(img)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Y = 0.299R + 0.587G + 0.114B

from PIL import Image
img = Image.open(imgfile).convert('LA')
img.save('greyscale.png')
