import ner_model
import inputData

class ner_utils(object):
    def __init__(self):
        self.model=ner_model.ner_model(False)

    def loadModel(self,path):
        self.model.loadModelForPredict(path)


    def predict(self,inputData):
        return  self.model.predictForInputData(inputData)

    def split(self,label,length):
        return  self.model.split(label,length)


def convertSimpleToInput(simple):
    inputWords=simple.inputWords
    labelsList=[]
    inputWordsList=[]
    for w in inputWords:
        labelsList.append(w.label)
        inputWordsList.append(w.text)
    return inputData.inputData("1",inputWordsList,labelsList)








