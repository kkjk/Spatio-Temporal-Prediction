from scipy.io import loadmat,savemat
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import Ridge
import sys

#: Read - Input and output from the csv files
relativeCurrDir = sys.argv[1]
inputFile = relativeCurrDir+"/ProcessedData/InputFeature.csv"
outputFile = relativeCurrDir+"/ProcessedData/OutputFeature.csv"
Input =pd.read_csv(inputFile, sep=',',header=None)
Output =pd.read_csv(outputFile, sep=',',header=None)

#: Model - Creation, train, score
linearTrainingModel = Ridge(alpha=1.0)
linearTrainingModel.fit(Input, Output)
r2ScoreTraining = linearTrainingModel.score(Input,Output)
print("Training R2 score is %s"%(r2ScoreTraining))

#: Save - the model to disk
filename = relativeCurrDir+'/Trained_Weights.sav'
pickle.dump(linearTrainingModel, open(filename, 'wb'))



