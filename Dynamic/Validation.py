
from scipy.io import loadmat,savemat
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import Ridge

#: Read - Input and output from the csv files
relativeCurrDir = sys.argv[1]
inputFile = relativeCurrDir+"/ProcessedData/InputValFeature.csv"
outputFile = relativeCurrDir+"/ProcessedData/OutputValFeature.csv"
ValidationDataInput = pd.read_csv(inputFile, sep=',',header=None)
ValidationDataOutput = pd.read_csv(outputFile, sep=',',header=None)
filename = relativeCurrDir+'/Trained_Weights.sav'

#: Model: Load trained weights
loaded_model = pickle.load(open(filename, 'rb'))

#: Check the score for validation
r2ScoreValidation = loaded_model.score(ValidationDataInput,ValidationDataOutput)

print("Validation r2 score of the linear regression model is %s"%(r2ScoreValidation))

