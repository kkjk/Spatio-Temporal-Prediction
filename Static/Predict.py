import os
from scipy.io import loadmat,savemat
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import time
import random
import pandas as pd
from sklearn.linear_model import Ridge
import sys

#: Variable Initializations

relativeCurrDir = sys.argv[1]
inputFile = relativeCurrDir+"/InputStaticFeature.csv"
outputFile = relativeCurrDir+"/OutputStaticFeature.csv"

DataInput = pd.read_csv(inputFile, sep=',',header=None)
DataOutput = pd.read_csv(outputFile, sep=',',header=None)
filename = relativeCurrDir+'/Trained_Weights.sav'

#: loading the pre trained weights onto the model
linearTrainingModel = Ridge(alpha=1.0)
linearTrainingModel.fit(DataInput, DataOutput)
r2ScoreTraining = linearTrainingModel.score(DataInput,DataOutput)

ImagesDir = relativeCurrDir+"/Images"
predictionDir = relativeCurrDir+"/PredictionRaw"

def prediction(username, num):
    """
    This function runs the prediction on a 1920x1200 images by reading the data
    from the mouse, cursor and the bounding box saliency maps. The corresponding 
    predicted images are stored in PredictionRaw folder of this directory

    Parameters:
        username (string): name of the user whose data will be processed
        num (int): Index of the interface
    """

    if not os.path.exists(predictionDir):
        os.mkdir(predictionDir)

    PredictedImg = Image.new('1',(1920, 1200), color=0)

    Bbox = Image.open(os.path.join(ImagesDir, str(username) + "_" + str(num)+ "_bbox.png"))
    Mouse = Image.open(os.path.join(ImagesDir, str(username) + "_" + str(num)+ "_mouse.png"))
    Cursor = Image.open(os.path.join(ImagesDir, str(username) + "_" + str(num)+ "_cursor.png"))

    for y in range(1200):
        for x in range(1920):
            temp = []
            temp.append(Mouse.getpixel(( x, y)))
            temp.append(Cursor.getpixel(( x, y)))
            temp.append(Bbox.getpixel(( x, y)))
            npTemp = np.array(temp).reshape(1, -1)
            fixPred = linearTrainingModel.predict(npTemp)
            if fixPred > 5:
                PredictedImg.putpixel((x,y), 1)
            else:
                PredictedImg.putpixel((x,y), 0)

    PredictedImg.save(os.path.join(predictionDir, str(username) + "_Prediction_"+str(num)+".png"))

#: Run predictions for users : Srikanth and Varun
#: To support more users, the corresponding input blurred images of mouse, cursor and bbox
#: needs to be added to the Image folder in this directory
prediction('Srikanth', 14)
prediction('Varun', 14)


