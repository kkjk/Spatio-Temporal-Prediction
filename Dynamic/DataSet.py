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
import sys
sys.path.append(os.path.abspath('.'))
from CommonUtils.classUtilities import *
from CommonUtils.funcUtilities import *


def createDataSet(userName, intfOneBBox, intfName, dimension, inputFile, outputFile):
    """
    This function reads the mat files for the mentioned user and collects
    all the positions for mouse, cursor and fixations w.r.t timestamps,
    interpolates, uniform samples the data points to collect 200 data points,
    normalize timestamp in range 0-1, read the data corresponding to the positions
    of the data samples, read data from random data points and save all the data
    into csv files - Input as well as Output csv files

    Parameters:
        userName (string): name of the user whose data will be processed
        intfOneBBox (InterfaceBbox): The instance of the class InterfaceBbox
            containing the data for the Bounding box for the corresponding
            interface
        intfName (string): The name of the selected interface
        dimension (int): Feature dimension can be calculated by 2*dimension + 1
        inputFile (string): File name which contains data for the input feature
    """
    mat = scipy.io.loadmat(userName)
    totalInterfaces = len(mat['guidata'][0])
    intfList = []
    for i in range(totalInterfaces):
        intfList.append(mat['guidata'][0][i][0][0][0][0])
    try:
        intfListIndex = intfList.index(intfName)
    except ValueError:
        print("No interface %s for user %s" % (intfName, userName))
        return

    ts_input_mouse = mat['guidata'][0][intfListIndex][0][0][2][0][0][0][0]
    ts_input_cursor = mat['guidata'][0][intfListIndex][0][0][2][0][0][3][0]
    ts_output_fixation = mat['guidata'][0][intfListIndex][0][0][3][0]

    #: Reading mouse, cursor and fixation data from the  mat file of the user
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

    #: Uniform sampling
    sampledNormList = random.sample(list(normalize(sampledTimestamp).keys()), k=200)
    sampledNormList.sort()

    #: dataList creation 
    normIntf1DataList, actualIntf1TimeList = culminateDataPoints(sampledNormList, normalize(sampledTimestamp),
                                                                 mouse_plot,
                                                                 cursor_plot, fixation_plot, mouse_dict, cursor_dict,
                                                                 fixation_dict, intfOneBBox)

    #: Reading data from the data positions read form the user file for mouse, cursor amd fixations 
    Inft1FeatureList, Inft1FixationFeatureList = readPositiveData(dimension, normIntf1DataList, actualIntf1TimeList,
                                                                  intfOneBBox)

    #: Reading data for data positions in the interface of 1900x1200 size 
    tempIntf1FeatureList, tempIntfFixationFeatureList = readNegativeData(dimension, normIntf1DataList,
                                                                         actualIntf1TimeList, intfOneBBox)
    Inft1FeatureList = Inft1FeatureList + tempIntf1FeatureList
    Inft1FixationFeatureList = Inft1FixationFeatureList + tempIntfFixationFeatureList

    #: Saving to a csv file 
    npInputArray = np.asarray(Inft1FeatureList)
    npOutputArray = np.asarray(Inft1FixationFeatureList)
    pd.DataFrame(npInputArray).to_csv(inputFile, mode='a', header=None, index=None)
    pd.DataFrame(npOutputArray).to_csv(outputFile, mode='a', header=None, index=None)


#####################################################################################################################################################

#: Intialization - variable initialization, read configuration, create folders if not created

relativeCurrDir = sys.argv[1]
dictOfInterfaceBbox = BoundingBox.getBoundingBoxInfo()
listOfInterfaces = getInterfacesList()
processedDataDir = relativeCurrDir+"/ProcessedData"
inputTrainFile = processedDataDir+"/InputFeature.csv"
outputTrainFile = processedDataDir+"/OutputFeature.csv"
inputValFile = processedDataDir+"/InputValFeature.csv"
outputValFile = processedDataDir+"/OutputValFeature.csv"
datasetDir = sys.argv[2]
try:
    feaDimension = int(sys.argv[3])
    print("Dimension attribute is %s"%(feaDimension))
except:
    print("Error: json file not parsed right, exiting")
    sys.exit(1)
    
if not os.path.exists(processedDataDir):
        os.mkdir(processedDataDir)

#: Data creation: Training

for userFile in getTrainingUserList():
    for sInterfaceName in listOfInterfaces:
        print("Interface: %s , User: %s"%(sInterfaceName,userFile))
        createDataSet(datasetDir+userFile, dictOfInterfaceBbox[sInterfaceName], sInterfaceName, feaDimension, inputTrainFile , outputTrainFile)
print("Training data created")

#: Data creation: Validation

for userFile in getValidationUserList():
    for sInterfaceName in listOfInterfaces:
        print("Interface: %s , User: %s"%(sInterfaceName,userFile))
        createDataSet(datasetDir+userFile, dictOfInterfaceBbox[sInterfaceName], sInterfaceName, feaDimension, inputValFile , outputValFile)
print("Validation data created")

