import torch
import torch.nn as nn
from utility import measure_time
from collections import defaultdict
import gc

DEVICE=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
WEIGHT=[1,1]

def generalDiceLoss(output,target):
    smooth=1.
    output=output[:,0]
    target=target[:,0]
    oflat = output.flatten()
    tflat = target.flatten()
    intersection = (oflat * tflat).sum()
    answer= 1 - ((2. * intersection + smooth) /
                  (oflat.sum() + tflat.sum() + smooth))
    return answer

def BinaryDiceLoss(output,targets,keys=None):
    smooth = 1.
    #answers=[]
    output=(output[:,0,:,:,:]+1-output[:,1,:,:,:])/2
    answers=[]
    keys=keys if keys else list(targets.keys())

    for key in keys:
        target=targets[key]
        target=target[:,0,:,:,:]>0.5
        oflat = output.flatten()
        tflat = target.flatten()
        intersection = (oflat * tflat).sum()
        #print(intersection,oflat.sum(),tflat.sum())

        answer= 1 - ((2. * intersection + smooth) /
                  ((oflat*oflat).sum() + (tflat*tflat).sum() + smooth))
        answers.append(answer)
        #answers=torch.cat([answers,answer])
        del answer
    return torch.mean(torch.stack(answers))

def BCE_with_DiceLoss(input,targets,keys=None):
    keys=keys if keys else list(targets.keys())
    dsl=DiceLoss(input,targets,keys=keys)
    bce=customBCELoss(input,targets,keys=keys)
    return (dsl+bce)/2

def normalDiceLoss(output,targets,key=None):
    smooth = 1.
    #answers=[]
    output=(output[:,0]+1-output[:,1])/2
    answers=[]
    if key:
        target=targets[key]
    else:
        target=list(targets.values())[0]
    target=target[:,0,:,:,:]>0.5
    oflat = output.flatten()
    tflat = target.flatten()
    intersection = (oflat * tflat).sum()

    answer= 1 - ((2. * intersection + smooth) /
                  (oflat.sum() + tflat.sum() + smooth))
    #answers.append(answer)
    return answer

def customBCELoss(output,targets,keys=None):
    loss_function=nn.BCELoss()
    answers=[]
    for key,target in targets.items():
        answers.append(loss_function(output[:,0],target[:,0]))
        
    return torch.mean(torch.stack(answers))

def DiceLoss(output,targets,keys=None):
    smooth = 1.
    #answers=[]
    output=output[:,0,:,:,:]
    answers=[]
    keys=keys if keys else list(targets.keys())

    for key in keys:
        target=targets[key]
        target=target[:,0,:,:,:]
        oflat = output.flatten()
        tflat = target.flatten()
        intersection = (oflat * tflat).sum()

        answer= 1 - ((2. * intersection + smooth) /
                  ((oflat*oflat).sum() + (tflat*tflat).sum() + smooth))
        answers.append(answer)
        #answers=torch.cat([answers,answer])
        del answer
    return torch.mean(torch.stack(answers))

def weightDiceLoss(output,targets,keys=None):
    smooth = 1.
    #answers=[]
    weightanswer=0
    for c in range(2):
        output=output[:,c,:,:,:]
        answers=[]
        keys=keys if keys else list(targets.keys())

        for key in keys:
            target=targets[key]
            target=target[:,c,:,:,:]
            oflat = output.flatten()
            tflat = target.flatten()
            intersection = (oflat * tflat).sum()

            answer= ((2. * intersection + smooth) /
                      ((oflat*oflat).sum() + (tflat*tflat).sum() + smooth))
            answers.append(answer)
            #answers=torch.cat([answers,answer])
        weightanswer+=torch.mean(torch.stack(answers))*WEIGHT[c]/sum(WEIGHT)
        return 1-weightanswer

class DSLManager():
    def __init__(self,keys=[]):
        self.smooth=1.
        self.sum_of_intersection=defaultdict(lambda :0)
        self.sum_flats=defaultdict(lambda :0)
        self.keys=keys
    
    def register(self,output,targets):
        output=output[:,0,:,:,:]
        if not self.keys:
            self.keys=list(targets.keys())
        
        oflat=torch.flatten(output)
        for key in self.keys:
            target=targets[key]
            target=target[:,0,:,:,:]
            tflat=torch.flatten(target)
            intersection = (oflat * tflat).sum()
            self.sum_of_intersection[key]+=intersection
            self.sum_flats[key]+=(oflat.sum()+tflat.sum())

            del tflat
        del oflat
        gc.collect()     

    def calculate(self):
        answers={}
        for key in self.keys:
            answer=1-((2. * self.sum_of_intersection[key] + self.smooth) /
              (self.sum_flats[key] + self.smooth))
            answers[key]=answer
        #print("sum_flat",self.sum_flats)
        #print("sum_intersection",self.sum_of_intersection)  
        #print("answers",answers)
        return sum(answers.values())/len(answers)

    def init_sum(self):
        self.sum_flats,self.sum_of_intersection=defaultdict(lambda: 0),defaultdict(lambda: 0)    

class BinaryDSLManager():
    def __init__(self,keys=[]):
        self.smooth=1.
        self.sum_of_intersection=defaultdict(lambda :0)
        self.sum_flats=defaultdict(lambda :0)
        self.keys=keys
    
    def register(self,output,targets):
        output=output[:,0,:,:,:]>0.5
        if not self.keys:
            self.keys=list(targets.keys())
        
        oflat=torch.flatten(output)
        for key in self.keys:
            target=targets[key]
            target=target[:,0,:,:,:]>0.5
            tflat=torch.flatten(target)
            intersection = (oflat * tflat).sum()
            self.sum_of_intersection[key]+=intersection
            self.sum_flats[key]+=((oflat*oflat).sum()+(tflat*tflat).sum())
            #print("intersection",intersection)
            #print("flat sum",(oflat.sum()+tflat.sum()))
            del tflat
        del oflat
        gc.collect()
        

    def calculate(self):
        answers={}
        for key in self.keys:
            answer=1-((2. * self.sum_of_intersection[key] + self.smooth) /
              (self.sum_flats[key] + self.smooth))
            answers[key]=answer
        #print("sum_flat",self.sum_flats)
        #print("sum_intersection",self.sum_of_intersection)  
        #print("answers",answers)
        return sum(answers.values())/len(answers)

    def init_sum(self):
        self.sum_flats,self.sum_of_intersection=defaultdict(lambda: 0),defaultdict(lambda: 0)    
