import os
import sys
from Configuration.ReadConfig import *

Service = sys.argv[1]
PortName = sys.argv[2]

masterConfig = ParseJson("./Configuration/Config.json")

try:
    masterConfig
except:
    sys.exit(1)


class Static:
    """
    This class acts as an interface to manage the functionalities of the static attention prediction
    which is a re implementation of the base model

    """
    def __init__(self):
        """
        Constructor of the class, initializes the actions supported by the static predictor

        """

        self.action = {"dataset": self.datasetCreation, "train": self.Train, "test": self.Test,
                       "evaluate": self.Evaluate, "quickeval": self.QuickEval}
        self.configMap = masterConfig.getStaticConfig()

    def datasetCreation(self):
        """
        Method that preforms the creation of dataset - training and validation for the static predictor
 
        """
        rawPredDir = "Static/ImagesRaw"
        predBlurDir = "Static/Images"
        #os.system('python3 CommonUtils/HeatMap.py %s %s'%(rawPredDir, predBlurDir))
        os.system('python3 Static/Static.py Static Dataset/')
        #os.system('python3 Static/GenerateGt.py Static Dataset/')
        rawgtDir = "Static/GroundTruthRaw"
        gtBlurDir = "Static/GroundTruth"
        #os.system('python3 CommonUtils/HeatMap.py %s %s'%(rawgtDir, gtBlurDir))



    def Train(self):
        """
        Method that calls the training functionality of the static predictor

        """
        os.system('python3 Static/Regression.py Static')

    def Test(self):
        """
        Method that calls the test functionality of the static predictor

        """
        rawPredDir = "Static/PredictionRaw"
        predBlurDir = "Static/Prediction"
        
        os.system('python3 Static/Predict.py Static')
        os.system('python3 CommonUtils/HeatMap.py %s %s'%(rawPredDir, predBlurDir))

    def Evaluate(self):
        """
        Method that can perform the evaluation after the test has been performed

        """
        predBlurDir = "Static/Prediction"
        gtDir = "Static/GroundTruth"
        resultFile = "Static"
        os.system('python3 CommonUtils/Saliency_metrics_evaluation.py %s %s %s'%(gtDir, predBlurDir, resultFile))

    def QuickEval(self):
        """
        Method for a quick evaluation, pre defined generated data

        """
        predBlurDir = "Static/QuickPred"
        gtDir = "Static/QuickGT"
        resultFile = "Static"
        os.system('python3 CommonUtils/Saliency_metrics_evaluation.py %s %s %s'%(gtDir, predBlurDir, resultFile))

    def checkPort(self, _portName):
        """
        Method that routs the action to a specific method

        """
        if _portName in self.action:
            self.action[_portName]()
        else:
            print("Input argument error. Please check")


class Dynamic:
    """
    This class acts as an interface to manage the functionalities of the dynamic attention prediction
    which is a re implementation of the base model

    """

    def __init__(self):
        """
        Constructor of the class, initializes the actions supported by the dynamic predictor

        """
        self.action = {"dataset": self.datasetCreation, "train": self.Train, "test": self.Test,
                       "evalLatest": self.Evaluate, "quickeval": self.QuickEval}	
        self.configMap = masterConfig.getDynamicConfig()

    def datasetCreation(self):
        """
        Method that preforms the creation of dataset - training and validation for the dynamic predictor

        """
        print("%s"%(str(self.configMap)))
        os.system('python3 Dynamic/DataSet.py Dynamic Dataset/ %s'%(self.configMap["FeatureDimension"]))

    def Train(self):
        """
        Method that calls the training functionality of the dynamic predictor

        """
        os.system('python3 Dynamic/Regression.py Dynamic')

    def Test(self):
        """
        Method that calls the test functionality of the dynamic predictor

        """
        intermediatePredDir = "Dynamic/Predictions_Intermediate"
        intermediateGtDir = "Dynamic/GroundTruth_Intermediate"
        rawPredDir = "Dynamic/Raw_Prediction"
        rawGtDir = "Dynamic/Raw_GroundTruth"
        predBlurDir = "Dynamic/Prediction"
        gtBlurDir = "Dynamic/GroundTruth"
        
        os.system('python3 Dynamic/Test.py Dynamic Dataset/ %s'%(self.configMap["FeatureDimension"]))
        os.system('python3 Dynamic/Thresholding.py %s %s'%(intermediatePredDir, rawPredDir))
        os.system('python3 Dynamic/Thresholding.py %s %s'%(intermediateGtDir, rawGtDir)) 
        os.system('python3 CommonUtils/HeatMap.py %s %s'%(rawPredDir, predBlurDir))
        os.system('python3 CommonUtils/HeatMap.py %s %s'%(rawGtDir, gtBlurDir))

    def Evaluate(self):
        """
        Method that can perform the evaluation after the test has been performed

        """
        rawPredDir = "Dynamic/Raw_Prediction"
        rawGtDir = "Dynamic/Raw_GroundTruth"
        predBlurDir = "Dynamic/Prediction"
        gtBlurDir = "Dynamic/GroundTruth"
        resultFile = "Dynamic"
        os.system('python3 CommonUtils/Saliency_metrics_evaluation.py %s %s %s'%(rawGtDir, predBlurDir, resultFile))

    def QuickEval(self):
        """
        Method for a quick evaluation, pre defined generated data

        """
        predBlurDir = "Dynamic/PredictionDefault"
        gtDir = "Dynamic/GroundTruthDefault"
        resultFile = "Dynamic"
        os.system('python3 CommonUtils/Saliency_metrics_evaluation.py %s %s %s'%(gtDir, predBlurDir, resultFile))

    def checkPort(self, _portName):
        """
        Method that routs the action to a specific method

        """
        if _portName in self.action:
            self.action[_portName]()
        else:
            print("Input argument error. Please check")

class LSTM_Static:
    """
    This class acts as an interface to manage the functionalities of the LSTM static attention prediction, 
    currently, mocked functionalites. 

    """

    def __init__(self):
        self.action = {"dataset": self.datasetCreation, "train": self.Train, "test": self.Test,
                       "evaluate": self.Evaluate}
        self.configMap = masterConfig.getLstmStaticConfig()

    def datasetCreation(self):
        print("Mock function")

    def Train(self):
        print("Mock function")

    def Test(self):
        print("Mock function")

    def Evaluate(self):
        print("Mock function")

    def checkPort(self, _portName):
        if _portName in self.action:
            self.action[_portName]()
        else:
            print("Input argument error. Please check")

class LSTM_Dynamic:
    """
    This class acts as an interface to manage the functionalities of the LSTM dynamic attention prediction, 
    currently, mocked functionalites. 

    """

    def __init__(self):
        self.action = {"dataset": self.datasetCreation, "train": self.Train, "test": self.Test,
                       "evaluate": self.Evaluate}
        self.configMap = masterConfig.getLstmDynamicConfig()

    def datasetCreation(self):
        print("Mock function")

    def Train(self):
        print("Mock function")

    def Test(self):
        print("Mock function")

    def Evaluate(self):
        print("Mock function")

    def checkPort(self, _portName):
        if _portName in self.action:
            self.action[_portName]()
        else:
            print("Input argument error. Please check")


ServiceDict = {"static": Static(), "dynamic": Dynamic(), "lstm_static": LSTM_Static(), "lstm_dynamic": LSTM_Dynamic() }

if Service in ServiceDict:
    ServiceDict[Service].checkPort(PortName)
else:
    print("Input argument error. Please check")


