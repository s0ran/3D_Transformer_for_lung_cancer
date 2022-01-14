import numpy as np
import torch 
from utility import measure_time
from collections import defaultdict
import gc

def BinaryDSC(output,targets,keys=[]):
    smooth=1.
    answers=[]
    output=output[:,0,:,:,:]>0.5
    keys=keys if keys else list(targets.keys())
    for key in keys:
        target=targets[key]
        target=target[:,0,:,:,:]>0.5
        oflat,tflat=torch.flatten(output),torch.flatten(target)
        intersection = (oflat * tflat).sum()
        answer=((2. * intersection + smooth) /
              (oflat.sum() + tflat.sum() + smooth))
        answers.append(answer)
    return sum(answers)/len(answers)

def DSC(output,targets,keys=[]):
    smooth=1.
    answers=[]
    output=output[:,0,:,:,:]
    keys=keys if keys else list(targets.keys())
    for key in keys:
        target=targets[key]
        target=target[:,0,:,:,:]
        oflat,tflat=torch.flatten(output),torch.flatten(target)
        intersection = (oflat * tflat).sum()
        answer=((2. * intersection + smooth) /
              (oflat.sum() + tflat.sum() + smooth))
        answers.append(answer)
    return sum(answers)/len(answers)

def generalDSC(output,target):
    smooth=1.
    output=output[:,0]
    target=target[:,0]
    oflat = output.flatten()
    tflat = target.flatten()
    intersection = (oflat * tflat).sum()
    answer= (2. * intersection + smooth) / (oflat.sum() + tflat.sum() + smooth)
    return answer

class BinaryDSCManager:
    def __init__(self,keys=[]):
        self.smooth=1.
        self.sum_of_intersection=defaultdict(lambda :0)
        #self.intersection_list=defaultdict(list)
        self.sum_flats=defaultdict(lambda :0)
        self.keys=keys
    
    def register(self,output,targets):
        output=output[:,0,:,:,:]>0.5
        oflat=torch.flatten(output)
        if not self.keys:
            self.keys=list(targets.keys())
        #print(targets.keys())
        for key in self.keys:
            target=targets[key]
            target=target[:,0,:,:,:]>0.5
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
            answer=((2. * self.sum_of_intersection[key] + self.smooth) /
              (self.sum_flats[key] + self.smooth))
            #print(answer)
            answers[key]=answer
        return sum(answers.values())/len(answers)

    def init_sum(self):
        self.sum_flats,self.sum_of_intersection=defaultdict(lambda: 0),defaultdict(lambda: 0)        


class DSCManager:
    def __init__(self,keys=[]):
        self.smooth=1.
        self.sum_of_intersection=defaultdict(lambda :0)
        #self.intersection_list=defaultdict(list)
        self.sum_flats=defaultdict(lambda :0)
        self.keys=keys
    
    def register(self,output,targets):
        output=output[:,0,:,:,:]
        oflat=torch.flatten(output)
        if not self.keys:
            self.keys=list(targets.keys())
        #print(targets.keys())
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
            answer=((2. * self.sum_of_intersection[key] + self.smooth) /
              (self.sum_flats[key] + self.smooth))
            #print(answer)
            answers[key]=answer
        return sum(answers.values())/len(answers)

    def init_sum(self):
        self.sum_flats,self.sum_of_intersection=defaultdict(lambda: 0),defaultdict(lambda: 0)        
