#!/usr/bin/env python
# coding: utf-8


import os
import scipy.io
from scipy.io import loadmat, savemat
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
from scipy.interpolate import interp1d
import pandas as pd
from numpy import savetxt
import pickle
import sys
sys.path.append(os.path.abspath('.'))
from CommonUtils.classUtilities import *
from CommonUtils.funcUtilities import *

SQUARE_SIZE =  50

def getRegion(X, Y, xMax, yMax, sqrSize):
    """
    This function computes the window for the region of interest

    Parameters:
        X (int): location on x axis
        Y (int): location on y axis
        xMax (int): max value on x axis
        yMax (int): max value on y axis
        sqrSize (int): window size

    Returns:
        list : computed coordinates [minX, maxX, minY, maxY]

    """
    regCoordinates = []
    regCoordinates.append((0, X - sqrSize)[(X - sqrSize) >= 0])
    regCoordinates.append((0, X + sqrSize)[(X + sqrSize) < xMax])
    regCoordinates.append((0, Y - sqrSize)[(Y - sqrSize) >= 0])
    regCoordinates.append((0, Y + sqrSize)[(Y + sqrSize) < yMax])
    return regCoordinates


def getRegionOfInterest(IntfDataList, time, dim):
    """
    This function computes the region of interest

    Parameters:
       IntfDataList (list): data list
        time (int): time index
        dim (int): The dimension can be computed by 2*dim+1

    Returns:
        list : computed coordinates of the region of interest [minX, maxX, minY, maxY]

    """
    XMinList = []
    XMaxList = []
    YMinList = []
    YMaxList = []
    regCoordinates = []

    for i in range(dim, len(IntfDataList) - dim):
        value = 0
        if IntfDataList[i].timestamp == time:
            for k in range(2 * dim):
                actualMouseX = IntfDataList[i - dim + k].mouse_data[0][0]
                actualMouseY = IntfDataList[i - dim + k].mouse_data[0][1]
                tempMouseCoordinateList = getRegion(actualMouseX, actualMouseY, 1920, 1200, SQUARE_SIZE)
                XMinList.append(tempMouseCoordinateList[0])
                XMaxList.append(tempMouseCoordinateList[1])
                YMinList.append(tempMouseCoordinateList[2])
                YMaxList.append(tempMouseCoordinateList[3])
                actualCursorX = IntfDataList[i - dim + k].cursor_data[0][0]
                actualCursorY = IntfDataList[i - dim + k].cursor_data[0][1]
                if actualCursorX == 2000:
                    actualCursorX = actualMouseX
                    actualCursorY = actualMouseY
                tempCursorCoordinateList = getRegion(actualCursorX, actualCursorY, 1920, 1200, SQUARE_SIZE)
                XMinList.append(tempCursorCoordinateList[0])
                XMaxList.append(tempCursorCoordinateList[1])
                YMinList.append(tempCursorCoordinateList[2])
                YMaxList.append(tempCursorCoordinateList[3])
            break
    regCoordinates.append(int(min(XMinList)))
    regCoordinates.append(int(max(XMaxList)))
    regCoordinates.append(int(min(YMinList)))
    regCoordinates.append(int(max(YMaxList)))
    return regCoordinates


def preprocess(userFIle, intfName, intfBBox, datasetdir):
    """
    This function reads the mat files for the mentioned user and collects
    all the positions for mouse, cursor and fixations w.r.t timestamps,
    interpolates, uniform samples the data points to collect 200 data points,
    normalize timestamp in range 0-1,

    Parameters:
        userFIle (string): name of the user whose data will be processed
        intfName (string): The name of the selected interface
        intfBBox (InterfaceBbox): The instance of the class InterfaceBbox
            containing the data for the Bounding box for the corresponding
            interface

    Returns:
        list : normalized data list
        list : actual list of timestamps
    """
    mat = scipy.io.loadmat(userFIle)
    totalInterfaces = len(mat['guidata'][0])

    intfList = []
    for i in range(totalInterfaces):
        intfList.append(mat['guidata'][0][i][0][0][0][0])

    intfListIndex = intfList.index(intfName)
    ts_input_mouse = mat['guidata'][0][intfListIndex][0][0][2][0][0][0][0]
    ts_input_cursor = mat['guidata'][0][intfListIndex][0][0][2][0][0][3][0]
    ts_output_fixation = mat['guidata'][0][intfListIndex][0][0][3][0]

    mouse_dict = {}

    for i in ts_input_mouse:
        if i['event'][0][0][0] == "move":
            if i['timestamp'][0][0][0][0] not in mouse_dict:
                mouse_dict[i['timestamp'][0][0][0][0]] = []
                mouse_dict[i['timestamp'][0][0][0][0]].append((i['x'][0][0][0][0], i['y'][0][0][0][0]))

    cursor_dict = {}

    for i in ts_input_cursor:
        if i['type'][0][0][0] == "caret":
            cursor_dict[i['timestamp'][0][0][0][0]] = []
            cursor_dict[i['timestamp'][0][0][0][0]].append((i['x'][0][0][0][0], i['y'][0][0][0][0]))

    fixation_dict = {}

    for i in ts_output_fixation:
        fixation_dict[i['timestamp'][0][0][0][0]] = []
        fixation_dict[i['timestamp'][0][0][0][0]].append((np.uint16(i['x'][0][0][0]), np.uint16(i['y'][0][0][0])))

    mouse_plot, cursor_plot, fixation_plot = getInterpolation(mouse_dict, cursor_dict, fixation_dict)

    sampledTimestamp = sampleTime(mouse_dict)

    sampledNormList = random.sample(list(normalize(sampledTimestamp).keys()), k=200)
    sampledNormList.sort()

    normIntf1DataList, actualIntf1TimeList = culminateDataPoints(sampledNormList, normalize(sampledTimestamp),
                                                                 mouse_plot, cursor_plot, fixation_plot, mouse_dict,
                                                                 cursor_dict, fixation_dict, intfBBox)

    return normIntf1DataList, actualIntf1TimeList


def predict(userFIle, intfBBox, timeIndex, dimension, dataList, timeList, predDir, gtDir):
    """
    Predicts the pixel value corresponding to the given input feature

    Parameters:
        userFIle (string): name of the user whose data will be processed
        intfBBox (InterfaceBbox): The instance of the class InterfaceBbox
            containing the data for the Bounding box for the corresponding
            interface
        timeIndex (int): Index of the timestamp
        dimension (int): 2d+1 the overall dimension
        dataList (int): list of the data points
        timeList (int): list of the time points
        predDir (string): directory to save for prediciton images
        gtDir (string): directory to save for ground truth images


    Returns:
        list : normalized data list
        list : actual list of timestamps
    """
    PredictedImg = Image.new('L', (1920, 1200))
    ActualImg = Image.new('L', (1920, 1200))
    predMaxList = []

    for i in range(len(dataList) - 1):
        value = 0
        if dataList[i].timestamp == timeList[timeIndex]:
            actualX = dataList[i].fixation_data[0][0]
            actualY = dataList[i].fixation_data[0][1]
            print("Timestamp is %s" % (timeList[timeIndex]))
            print("Actual value of fixation %s and %s " % (actualX, actualY))
            break

    regionOfInterest = getRegionOfInterest(dataList, timeList[timeIndex], dimension)
    print("The region of interest %s, %s, %s, %s" % (
    regionOfInterest[0], regionOfInterest[1], regionOfInterest[2], regionOfInterest[3]))
    print(regionOfInterest)
    for y in range(regionOfInterest[2], regionOfInterest[3]):
        for x in range(regionOfInterest[0], regionOfInterest[1]):
            feaList = getFeatureList(dataList, timeList[timeIndex], x, y, dimension, intfBBox)
            np_array = np.asarray(feaList)
            predictedVal = int(loaded_model.predict(np_array.reshape(1, -1)))
            tup = (x, y)
            if predictedVal >= 250:
                predMaxList.append(tup)
            PredictedImg.putpixel(tup, predictedVal)
            actualVal = getFixationFeatureList(dataList, timeList[timeIndex], x, y)
            ActualImg.putpixel(tup, actualVal)
            print("Current pixel x: %s and y: %s" % (x, y))

    print("The values above threshold are: ")
    for j in predMaxList:
        print(j)

    predFileName = predDir + "/" + userFIle.split(".")[0] + "_" + "Prediction" + "_" + str(timeIndex) + ".png"
    actualFileName = gtDir + "/" + userFIle.split(".")[0] + "_" + "Actual" + "_" + str(timeIndex) + ".png"

    PredictedImg.save(predFileName)
    ActualImg.save(actualFileName)


#: Intialization - variable initialization, read configuration, create folders if not created


relativeCurrDir = sys.argv[1]
datasetDir = sys.argv[2]
confMap = sys.argv[3]

try:
    filename = relativeCurrDir+'/Trained_Weights.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    dictOfInterfaceBbox = BoundingBox.getBoundingBoxInfo()
    listofInterfaces = ["Blogger_5"]
    listOfUsersMatFiles = ["Srikanth.mat"]
    timeIntervalMin = 10
    timeIntervalMax = 30
    timeIntervalStep = 5
    dimension = int(confMap)
    predictionDir = relativeCurrDir+"/Predictions_Intermediate"
    gtDir = relativeCurrDir+"/GroundTruth_Intermediate"
    if not os.path.exists(predictionDir):
        os.mkdir(predictionDir)
    if not os.path.exists(gtDir):
        os.mkdir(gtDir)

except:
    print("Error: json file not parsed right, exiting")
    sys.exit(1)

print("Prediction %s" % (timeIntervalMin))

for userFile in listOfUsersMatFiles:
    for interfaceName in listofInterfaces:
        _dataList, _timeList = preprocess(datasetDir + userFile, interfaceName, dictOfInterfaceBbox[interfaceName],
                                          datasetDir)
        for t in range(timeIntervalMin, timeIntervalMax, timeIntervalStep):
            print("Prediction started for time index %s" % (t))
            #: Prediction for a specific user, specific interface, specific time
            predict(userFile, dictOfInterfaceBbox[interfaceName], t, dimension , _dataList, _timeList, predictionDir , gtDir )

print("Predictions generation successful")


