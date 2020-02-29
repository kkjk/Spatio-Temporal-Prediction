import os
import numpy as np
from PIL import Image
import pandas as pd
import cv2 as cv
import sys

# This script is used to threshold the image to its (max pixel value - 1)
# The images from the input directory are read, thresholded and the resulting
# image is saved in the output direcory with the same file name
# This file is mainly used by the orchestrator after the test prediction
# The input directory is deleted at the end to remove the clutter


imdir = sys.argv[1]
outdir = sys.argv[2]
if not os.path.exists(outdir):
  os.makedirs(outdir)

for sFile in os.listdir(imdir):
  
  if sFile.endswith(".png"):
    path = os.path.join(imdir, sFile)
    print(path)
      
    img	 = cv.imread(path, cv.IMREAD_GRAYSCALE)
    numImg =  np.asarray(Image.open(path))
    biggest = np.amax(numImg)
    retActual,threshActual = cv.threshold(img,biggest-1,255,cv.THRESH_BINARY)
    cv.imwrite(os.path.join(outdir, sFile), threshActual)
    os.remove(os.path.join(imdir, sFile))

os.rmdir(imdir)



