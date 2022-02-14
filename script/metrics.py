from email.policy import default
import numpy as np
import torch 
from utility import measure_time
from collections import defaultdict
import gc

WEIGHT=[1,1]

def BinaryDSC(output,target):
    smooth=1.
    answers=[]
    output=output[:,0,:,:,:]>0.5
    target=target[:,0,:,:,:]>0.5
    oflat,tflat=torch.flatten(output),torch.flatten(target)
    intersection = (oflat * tflat).sum()
    answer=((2. * intersection + smooth) /
          (oflat.sum() + tflat.sum() + smooth))
    return answer

def DSC(output,target):
    smooth=1.
    output=output[:,0,:,:,:]
    target=target[:,0,:,:,:]
    oflat,tflat=torch.flatten(output),torch.flatten(target)
    intersection = (oflat * tflat).sum()
    answer=(2. * intersection + smooth) / ((oflat*oflat).sum() + (tflat*tflat).sum() + smooth)
    return answer

def weightDSC(outputs,targets):
    smooth = 1.
    weightanswer=0
    for c in range(2):
        channel=outputs[:,c,:,:,:]
        target=targets[:,c,:,:,:]
        oflat = channel.flatten()
        tflat = target.flatten()
        intersection = (oflat * tflat).sum()
        answer= (2. * intersection + smooth) / ((oflat*oflat).sum() + (tflat*tflat).sum() + smooth)
        weightanswer+=answer*WEIGHT[c]
    return weightanswer/sum(WEIGHT)

def BinaryweightDSC(outputs,targets):
    smooth = 1.
    weightanswer=0
    for c in range(2):
        channel=outputs[:,c,:,:,:]>0.5
        target=targets[:,c,:,:,:]>0.5
        oflat = channel.flatten()
        tflat = target.flatten()
        intersection = (oflat * tflat).sum()
        answer= (2. * intersection + smooth) / ((oflat*oflat).sum() + (tflat*tflat).sum() + smooth)
        weightanswer+=answer*WEIGHT[c]
    return weightanswer/sum(WEIGHT)

class BinaryDSCManager:
    def __init__(self):
        self.smooth=1.
        self.sum_of_intersection=0
        self.sum_flats=0  
    
    def register(self,output,target):
        output=output[:,0,:,:,:]>0.5
        oflat=torch.flatten(output)
        target=target[:,0,:,:,:]>0.5
        tflat=torch.flatten(target)
        intersection = (oflat * tflat).sum()
        self.sum_of_intersection+=intersection
        self.sum_flats+=((oflat*oflat).sum()+(tflat*tflat).sum())
        del tflat,oflat
        gc.collect()

    def calculate(self):
        answer=((2. * self.sum_of_intersection + self.smooth) /
              (self.sum_flats + self.smooth))
        return answer

    def init_sum(self):
        self.sum_flats,self.sum_of_intersection=defaultdict(lambda: 0),defaultdict(lambda: 0)        

class BinaryweightDSCManager:
    def __init__(self):
        self.smooth=1.
        self.sum_of_intersection=defaultdict(lambda : 0)
        self.sum_flats=defaultdict(lambda : 0)
        
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
            answer+=((2. * self.sum_of_intersection[c] + self.smooth) *WEIGHT[c]/
              (self.sum_flats[c] + self.smooth))          
        return answer/sum(WEIGHT)

    def init_sum(self):
        self.sum_flats,self.sum_of_intersection=0,0    

class DSCManager:
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
        torch.cuda.empty_cache()

    def calculate(self):
        answer=((2. * self.sum_of_intersection + self.smooth) /
          (self.sum_flats + self.smooth))
        return answer

    def init_sum(self):
        self.sum_flats,self.sum_of_intersection=0,0       



if __name__=="__main__":
    #a=torch.rand(1,1,320,512,512)
    #a=torch.cat([a,1-a],dim=1)
    #b=torch.rand(1,1,320,512,512)
    #bina=a>0.5
    #print("DSC(a,a)=",DSC(a,{"a":a}))
    #print("DSC(bina,bina)=",DSC(bina,{"a":bina}))
    #print("DSC(a,bina)=",DSC(a,{"a":bina}))
    #print("BinaryDSC(a,a)=",BinaryDSC(a,{"a":a}))
    #print("weightDSC(a,bina)=",weightDSC(a,{"a":bina}))
    #print("weightDSC(bina,bina)=",weightDSC(bina,{"a":bina}))
    a=torch.rand(1,1,320,512,512)
    b=torch.rand(1,1,320,512,512)
    z=torch.zeros(1,1,320,512,512)
    a=torch.cat([a,1-a],dim=1)
    b=torch.cat([b,1-b],dim=1)
    z=torch.cat([z,1-z],dim=1)
    print(b.min(),b.max(),b.sum())
    bsc_man=BinaryweightDSCManager()

    bsc_man.register(z,{"a":b})
    bsc_man.register(z,{"a":b})
    loss=bsc_man.calculate()
    print(loss)
