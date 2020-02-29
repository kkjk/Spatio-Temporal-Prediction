import os
import scipy
import scipy.io
from scipy.io import loadmat,savemat
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
import pandas as pd
from scipy.ndimage import gaussian_filter
import sys
sys.path.append(os.path.abspath('.'))
from CommonUtils.classUtilities import *
from CommonUtils.funcUtilities import *

relativeCurrDir = sys.argv[1]
ImagesDir = relativeCurrDir+"/Images"
gtDir = relativeCurrDir+"/GroundTruthRaw/"

def generateImage(imagePath, tupList, fileName):
    genImg = Image.new('L', (1920, 1200))
    xRange = range(0, 1920)
    yRange = range(0, 1200)
    for tupPos in tupList:
        if tupPos[0] in xRange:
            if tupPos[1] in yRange:
                genImg.putpixel(tupPos, 255)
    resultFile = imagePath+fileName
    genImg.save(resultFile)

datasetDir = sys.argv[2]
listofUsersVal = ["Srikanth.mat", "Varun.mat"]
listOfInterfaces = ["Blogger_5"]

#listofUsersVal = ["Alice.mat", "Charlotte.mat", "Irina.mat", "Konstantin.mat", "Mike.mat", "Russa.mat"]

for userFile in listofUsersVal:
    for sInterfaceName in listOfInterfaces:
        print("Interface: %s , User: %s"%(sInterfaceName,userFile))
        lMouse, lCursor, lFixation = readDataSet(userFile, sInterfaceName, datasetDir)
        if not lFixation:
            print("No data to process")
        else:
            name = userFile.split(".")[0] + "_" + "Actual" + "_" + str(14) + ".png"
            generateImage(gtDir, lFixation, name )
