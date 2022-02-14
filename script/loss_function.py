import torch
import torch.nn as nn
from utility import measure_time
from collections import defaultdict
import gc

DEVICE=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
WEIGHT=[1,1]

def BinaryDiceLoss(output,target):
    smooth = 1.
    #answers=[]
    output=output[:,0,:,:,:]>0.5
    answers=[]

    target=target[:,0,:,:,:]>0.5
    oflat = output.flatten()
    tflat = target.flatten()
    intersection = (oflat * tflat).sum()
    #print(intersection,oflat.sum(),tflat.sum())
    answer= 1 - ((2. * intersection + smooth) /
              ((oflat*oflat).sum() + (tflat*tflat).sum() + smooth))

    #answers=torch.cat([answers,answer])
    return answer

def BCE_with_DiceLoss(input,target):
    #keys=keys if keys else list(target.keys())
    dsl=DiceLoss(input,target)
    bce=customBCELoss(input,target)
    return (dsl+bce)/2

def customBCELoss(output,target):
    loss_function=nn.BCELoss()
    answer=loss_function(output,target)
        
    return answer

def DiceLoss(output,target):
    smooth = 1.
    output=output[:,0,:,:,:]

    target=target[:,0,:,:,:]
    oflat = output.flatten()
    tflat = target.flatten()
    intersection = (oflat * tflat).sum()
    answer= 1 - ((2. * intersection + smooth) /
              ((oflat*oflat).sum() + (tflat*tflat).sum() + smooth))
    #answers=torch.cat([answers,answer])
    return answer

def weightDiceLoss(outputs,targets):
    smooth = 1.
    #answers=[]
    weightanswer=0
    for c in range(2):
        channel=outputs[:,c,:,:,:]
        target=targets[:,c,:,:,:]
        oflat = channel.flatten()
        tflat = target.flatten()
        intersection = (oflat * tflat).sum()
        answer= ((2. * intersection + smooth) /
                  ((oflat*oflat).sum() + (tflat*tflat).sum() + smooth))
        weightanswer+=answer*WEIGHT[c]/sum(WEIGHT)
    return 1-weightanswer

def invDiceLoss(output,target):
    smooth = 1.
    output=output[:,0,:,:,:]

    target=target[:,0,:,:,:]
    oflat = output.flatten()
    tflat = target.flatten()
    intersection = (oflat * tflat).sum()
    dsc= ((2. * intersection + smooth) /
              ((oflat*oflat).sum() + (tflat*tflat).sum() + smooth))
    answer=1.01/(dsc+0.01)
    return answer

def invweightDiceLoss(output,target):
    wdsc=1-weightDiceLoss(output,target)
    return 1.01/(wdsc+0.01)

def MSELoss(output,target):
    loss_function=nn.MSELoss()
    loss=loss_function(output*10,target*10)
    return loss


class DSLManager():
    def __init__(self):
        self.smooth=1.
        self.sum_of_intersection=0
        self.sum_flats=0
    
    def register(self,output,target):
        output=output[:,0,:,:,:]
        oflat=torch.flatten(output)
        target=target[:,0,:,:,:]
        tflat=torch.flatten(target)
        intersection = (oflat * tflat).sum()
        self.sum_of_intersection+=intersection
        self.sum_flats+=((oflat*oflat).sum()+(tflat*tflat).sum())
        del tflat,oflat
        gc.collect()     

    def calculate(self):
        answer=1-((2. * self.sum_of_intersection[key] + self.smooth) /
          (self.sum_flats[key] + self.smooth))
        return answer

    def init_sum(self):
        self.sum_flats,self.sum_of_intersection=0,0  

class BinaryDSLManager():
    def __init__(self,keys=[]):
        self.smooth=1.
        self.sum_of_intersection=0
        self.sum_flats=0
        self.keys=keys
    
    def register(self,output,target):
        output=output[:,0,:,:,:]>0.5
        oflat=torch.flatten(output)
        target=target[:,0,:,:,:]>0.5
        tflat=torch.flatten(target)
        intersection = (oflat * tflat).sum()
        self.sum_of_intersection+=intersection
        self.sum_flats+=((oflat*oflat).sum()+(tflat*tflat).sum())
        #print("intersection",intersection)
        #print("flat sum",(oflat.sum()+tflat.sum()))
        del tflat,oflat
        gc.collect()
        

    def calculate(self):
        answers={}
        for key in self.keys:
            answer=1-((2. * self.sum_of_intersection[key] + self.smooth) /
              (self.sum_flats[key] + self.smooth))
            answers[key]=answer
        return sum(answers.values())/len(answers)

    def init_sum(self):
        self.sum_flats,self.sum_of_intersection=defaultdict(lambda: 0),defaultdict(lambda: 0)    

class BinaryweightDSLManager():
    def __init__(self,keys=[]):
        self.smooth=1.
        self.sum_of_intersection=defaultdict(lambda: 0)
        self.sum_flats=defaultdict(lambda: 0)
    
    def register(self,outputs,targets):
        for c in range(2):
            output=outputs[:,c,:,:,:]>0.5
            oflat=torch.flatten(output)
            target=targets[:,c,:,:,:]>0.5
            tflat=torch.flatten(target)
            intersection = (oflat * tflat).sum()
            self.sum_of_intersection[c]+=intersection
            self.sum_flats[c]+=((oflat*oflat).sum()+(tflat*tflat).sum())
            del tflat,oflat
            gc.collect()
            torch.cuda.empty_cache()
        

    def calculate(self):
        answer=0
        for c in range(2):
            answer+=1-((2. * self.sum_of_intersection[c] + self.smooth)*WEIGHT[c]/
              (self.sum_flats[c] + self.smooth))
        #print(answers)
        return answer/sum(WEIGHT)

    def init_sum(self):
        self.sum_flats,self.sum_of_intersection=defaultdict(lambda: 0),defaultdict(lambda: 0) 

if __name__=="__main__":
    a=torch.rand(1,1,320,512,512)
    b=torch.rand(1,1,320,512,512)
    z=torch.zeros(1,1,320,512,512)
    a=torch.cat([a,1-a],dim=1)
    b=torch.cat([b,1-b],dim=1)
    z=torch.cat([z,1-z],dim=1)
    #print(b.min(),b.max(),b.sum())
    #bsl_man=BinaryweightDSLManager()
    #bsl_man.register(z,{"a":b})
    #bsl_man.register(z,{"a":b})
    #loss=bsl_man.calculate()

    loss=customBCELoss(z,b)
    print(loss)

