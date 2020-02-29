#!/usr/bin/env python
# coding: utf-8


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
datasetdir = sys.argv[2]

#####################################################################################################################################################


class Bbox:
    def __init__(self, name,minX,minY,maxX,maxY):
        self.name = name
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY

class InterfaceOneBBox:
    def __init__(self, intfName):
        self.bboxList = []
        self.Name = intfName
    
    def addBbox(self, bboxInstance):
        self.bboxList.append(bboxInstance)

    def whichBbox(self, x, y):
        for bboxIns in self.bboxList:
            if(x > bboxIns.minX and x < bboxIns.maxX):
                if(y> bboxIns.minY and y < bboxIns.maxY ):
                    return bboxIns.name
        return "None"

#####################################################################################################################################################

def createDataSet(intfName, taskName, inputFile, outputFile):

    ImageDir = relativeCurrDir+"/Images/"
    mat = scipy.io.loadmat(datasetdir+intfName)
    totalInterfaces = len(mat['guidata'][0])

    intfList = []
    for i in range(totalInterfaces):
        intfList.append(mat['guidata'][0][i][0][0][0][0])

    try:
        intfListIndex = intfList.index(taskName)
        print(intfListIndex)
    except ValueError:
        print("No interface %s for user %s" % (taskName, intfName))
        return

    ts_input_mouse = mat['guidata'][0][intfListIndex][0][0][2][0][0][0][0]
    ts_input_cursor = mat['guidata'][0][intfListIndex][0][0][2][0][0][3][0]
    ts_output_fixation = mat['guidata'][0][intfListIndex][0][0][3][0]

    
    mouse_list = []

    for i in ts_input_mouse: 
        if i['event'][0][0][0] == "move":  
                mouse_list.append((i['x'][0][0][0][0],i['y'][0][0][0][0]))

    cursor_list = []

    for i in ts_input_cursor: 
        if i['type'][0][0][0] == "caret":
            cursor_list.append((i['x'][0][0][0][0],i['y'][0][0][0][0]))

    fixation_list = []

    for i in ts_output_fixation: 
        fixation_list.append((np.uint16(i['x'][0][0][0]), np.uint16(i['y'][0][0][0])))

    positive_list = mouse_list + cursor_list + fixation_list

    masterPositiveSpatialList = []
    masterFixationList = []


    mouseImage =  ImageDir + intfName[: -4] + "_"+ str(intfListIndex+1) +"_mouse"+".png"
    cursorImage =  ImageDir + intfName[: -4] + "_"+ str(intfListIndex+1)+"_cursor"+".png"
    fixationImage =  ImageDir + intfName[: -4] + "_"+ str(intfListIndex+1)+"_fixation"+".png"
    bboxImage =  ImageDir + intfName[: -4] + "_"+ str(intfListIndex+1)+"_bbox"+".png"


################### read positive data ############################

    for posXY in positive_list:
        tempList = []

        im = Image.open(mouseImage)
        tempList.append(im.getpixel((int(posXY[0]), int(posXY[1]))))

        im = Image.open(cursorImage)
        # print((int(posXY[0]), int(posXY[1])))
        tempList.append(im.getpixel((int(posXY[0]), int(posXY[1]))))

        im = Image.open(bboxImage)
        tempList.append(im.getpixel((int(posXY[0]), int(posXY[1]))))

        im = Image.open(fixationImage)
        masterFixationList.append(im.getpixel((int(posXY[0]), int(posXY[1]))))

        masterPositiveSpatialList.append(tempList)

    xAxis = list(range(1920))
    yAxis = list(range(1200))

################### read negative data ############################

    randXAxis = random.sample(xAxis,k=200)
    randYAxis = random.sample(yAxis,k=200)

    for i in range(200):
            tempList = []
            im = Image.open(mouseImage)
            tempList.append(im.getpixel((randXAxis[i], randYAxis[i])))

            im = Image.open(cursorImage)
            tempList.append(im.getpixel((randXAxis[i], randYAxis[i])))

            im = Image.open(bboxImage)
            tempList.append(im.getpixel((randXAxis[i], randYAxis[i])))

            im = Image.open(fixationImage)
            masterFixationList.append(im.getpixel((randXAxis[i], randYAxis[i])))

            masterPositiveSpatialList.append(tempList)

    ## Saving to a csv file ##

    npInputArray =  np.asarray(masterPositiveSpatialList)
    npOutputArray = np.asarray(masterFixationList)

    pd.DataFrame(npInputArray).to_csv(inputFile,  mode='a', header=None, index=None)
    pd.DataFrame(npOutputArray).to_csv(outputFile,  mode='a', header=None, index=None)
    



#####################################################################################################################################################

inputFile = relativeCurrDir+"/InputStaticFeature.csv"
outputFile = relativeCurrDir+"/OutputStaticFeature.csv"

listOfUsersMatFiles = ["Alice.mat", "Charlotte.mat", "Irina.mat", "Konstantin.mat", "Mike.mat"]
#listOfUsersMatFiles = ["Srikanth.mat", "Varun.mat"]

for userFile in listOfUsersMatFiles:
    createDataSet(userFile, 'Blogger_5', inputFile, outputFile)

print("Created")


