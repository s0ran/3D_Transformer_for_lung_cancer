import time
from functools import wraps
import os
import pandas as pd
import torch

def measure_time(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start=time.time()
        res=func(*args,**kwargs)
        end=time.time()
        print(f"{func.__name__} function took {int((end-start)/60//60)}[h] {int((end-start)//60%60)}[min] {int(end-start)%24%60}[s] == {end-start}[s]",flush=True)
        return res
    return wrapper 

class OutputLogger():
    def __init__(self,filepath):
        self.filepath=filepath

    def add(self,input,predict,label,metadata=""):
        with open(self.filepath,"a") as f:
            print(metadata,file=f)
            f.write(f"input:\n{input}\n")
            f.write(f"predict:\n{predict}\n")
            f.write(f"label:\n{label}\n")

def make_testcase():
    a=torch.rand(48,48,48,requires_grad=True,device="cuda:0")
    b=torch.rand(16,16,16,device="cuda:0")
    input=torch.stack([torch.stack([a,1-a])])
    targets={"b":torch.stack([torch.stack([b,1-b])])}
    return a,input,targets

def torch_info():
    print("torch version: ",torch.__version__)
    print("cuda version : ",torch.version.cuda)
    print("cudnn version: ",torch.backends.cudnn.version()) 