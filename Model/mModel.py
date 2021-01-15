from Model.model import Model, ModelUnet


class MVModel(object):
    def __init__(self,config):
        self.config = config
        self.mvList=[]

        self.View_num = config['View_num']

    def getModel(self):
        for i in range(self.View_num):
            self.mvList.append(Model(self.config))
        return  self.mvList

class MVUnetModel(object):
    def __init__(self,config):
        self.config = config
        self.mvList=[]

        self.View_num = config['View_num']

    def getModel(self):
        for i in range(self.View_num):
            self.mvList.append(ModelUnet(self.config))
        return  self.mvList
