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
import pickle
from sklearn.linear_model import Ridge
import sys

relativeCurrDir = sys.argv[1]
Input=pd.read_csv(relativeCurrDir+"/InputStaticFeature.csv", sep=',',header=None)
Output=pd.read_csv(relativeCurrDir+"/OutputStaticFeature.csv", sep=',',header=None)


#: Model - Creation, train, score
linearTrainingModel = Ridge(alpha=1.0)
linearTrainingModel.fit(Input, Output)
r2ScoreTraining = linearTrainingModel.score(Input,Output)
print("Training R2 score is %s"%(r2ScoreTraining))


#: Save - the model to disk
filename = relativeCurrDir+'/Trained_Weights.sav'
pickle.dump(linearTrainingModel, open(filename, 'wb'))
