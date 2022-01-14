import torch 
import torch.nn as nn
from torch.optim  import Adam
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter

import os
import sys
from tqdm import tqdm
import datetime
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile
import gc

from dataset import CONFIG_DATASET, LungCTFullImageDataset
from loss_function import DiceLoss, DSLManager,BinaryDSLManager,BinaryDiceLoss, generalDiceLoss,normalDiceLoss,customBCELoss,BCE_with_DiceLoss
from metrics import DSC, DSCManager,BinaryDSC,BinaryDSCManager
from utility import OutputLogger, measure_time
from model import ThreeDimensionalTransformer,load_model,save_model
from config import CONFIG
import json

now=datetime.datetime.now()
CONFIG_PATH=CONFIG["PATH"]
CONFIG_PRETRAIN=CONFIG["Pretraining"]
Datasets={
    0:"LungCTFullImageDataset",
    }
Keyslist={
    0:["label"],
    }
Models={
    0:"ThreeDimensionalTransformer",
}
DATASET_ID=CONFIG_PRETRAIN.getint("DATASET_ID")
DATASET=Datasets[DATASET_ID]
MODEL_TYPE=Models[CONFIG_PRETRAIN.getint("MODEL_ID")]
LABEL_KEYS=Keyslist[DATASET_ID]
BATCH_SIZE=CONFIG_PRETRAIN.getint("BATCH_SIZE")
DEVICE=torch.device(CONFIG_PRETRAIN.get("DEVICE")) if torch.cuda.is_available() else torch.device("cpu")
DEBUG=False
DEBUG2=False
ID=f"{str(now.date())}-{now.hour}-{now.minute}"
PRETRAIN_MODEL_PATH=os.path.join(CONFIG_PATH["PRETRAIN_MODELPATH"],f"{ID}.pt")
OUTPUTLOG_PATH=os.path.join(CONFIG_PATH.get("PRETRAIN_LOGPATH"),f"{ID}.txt")
EPOCH=CONFIG_PRETRAIN.getint("EPOCH")
LEARNINGLATE=CONFIG_PRETRAIN.getfloat("LEARNINGLATE")
WRITER=SummaryWriter()
LOGGER=OutputLogger(OUTPUTLOG_PATH) if CONFIG_PRETRAIN.getboolean("OUTPUT_LOG") else None
LOG_COUNT=10
TQDM_ENABLED=CONFIG_PRETRAIN.getboolean("TQDM_ENABLED")
NUM_WORKERS=CONFIG_PRETRAIN.getint("NUM_WORKERS")
Loss_for_one_patch_Index=0
DSC_on_evaluation_Index=0
DSC_Avg_for_one_epoch_Index=0
DSC_on_training_Index=0

#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = False


@measure_time
#@profile
def train_one_loader(model,optimizer,loss_function,trainloader):
    global Loss_for_one_patch_Index
    dscmanager=DSCManager(keys=LABEL_KEYS)
    dslmanager=DSLManager(keys=LABEL_KEYS)
    for data,labels in tqdm(trainloader,disable=not TQDM_ENABLED):
        if data[:,0].max()==0:
            del data,labels
            torch.cuda.empty_cache()
            gc.collect()
            continue
        labels={key:value.to(DEVICE) for key,value in labels.items()}
        data=data.to(DEVICE)
        optimizer.zero_grad()
        pre=model(data)
        dscmanager.register(pre,labels)
        dslmanager.register(pre,labels)
        loss=loss_function(pre,labels)
        #print(f"loss{loss}")
        """if loss>0.9 and LOGGER and 2500>Loss_for_one_patch_Index>2000:
            LOGGER.add(data,pre,labels)"""
        loss.backward()
        optimizer.step()
        WRITER.add_scalar("process/loss_for_one_patch",loss,Loss_for_one_patch_Index)
        Loss_for_one_patch_Index+=1
        del data,loss,pre,labels
        torch.cuda.empty_cache()
        gc.collect()
        if DEBUG:
            break
    dsc=dscmanager.calculate()
    loss=dslmanager.calculate()
    del dscmanager
    gc.collect()
    return model,dsc,loss

@measure_time
#@profile
def validate_one_loader(model,valloader):
    dsc_man=DSCManager(keys=LABEL_KEYS)
    dsl_man=DSLManager(keys=LABEL_KEYS)
    for data,labels in tqdm(valloader,disable=not TQDM_ENABLED):
        data=data.to(DEVICE)
        labels={key:value.to(DEVICE) for key,value in labels.items()}
        pre=model(data)
        dsc_man.register(pre,labels)
        dsl_man.register(pre,labels)
        del data,labels,pre
        torch.cuda.empty_cache()
        gc.collect()
        if DEBUG:
            break
    dsc=dsc_man.calculate()
    loss=dsl_man.calculate()
    del dsc_man,dsl_man
    gc.collect()
    #print(dsc,loss)
    return dsc,loss

@measure_time
def pretrain(model,traindata,valdata):
    global DSC_on_evaluation_Index,DSC_Avg_for_one_epoch_Index,DSC_on_training_Index
    loss_function=DiceLoss
    optimizer=Adam(model.parameters(),lr=LEARNINGLATE)
    #print(id(model))
    for epoch in tqdm(range(1,EPOCH+1),disable=not TQDM_ENABLED):
        print(f"------------EPOCH {epoch}/{EPOCH}------------")
        model.train()
        for _,patch_provider in tqdm(traindata,disable=not TQDM_ENABLED):
            trainloader=DataLoader(patch_provider,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)

            model,dsc,loss=train_one_loader(model,optimizer,loss_function,trainloader)
            
            WRITER.add_scalar("process/DSC_on_training",dsc,DSC_on_training_Index)
            DSC_on_training_Index+=1
            print("Image Dice Similarity Coefficient",dsc)
            print("finish training")
            print(f"whole loss {loss}")
            if DEBUG or DEBUG2:
                break
        model.eval()
        print("evaluation")
        with torch.no_grad():  
            loss_list=[]
            dsc_list=[]
            for _,patchprovider in tqdm(valdata,disable=not TQDM_ENABLED):
                #print(patchprovider)
                valloader=DataLoader(patchprovider,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
                dsc,loss=validate_one_loader(model,valloader)
                WRITER.add_scalar("process/DSC_on_evaluation",dsc,DSC_on_evaluation_Index)
                DSC_on_evaluation_Index+=1
                #print(dsc,loss)
                loss_list.append(loss)
                dsc_list.append(dsc)
                if DEBUG or DEBUG2:
                    break
                
            loss_list=torch.stack(loss_list)
            dsc_list=torch.stack(dsc_list)
            loss_avg=torch.mean(loss_list)
            loss_std=torch.std(loss_list)
            dsc_avg=torch.mean(dsc_list)
            dsc_std=torch.std(dsc_list)
            print("DSC average : ",dsc_avg)
            WRITER.add_scalar("result/DSC_Avg_for_one_epoch",dsc_avg,DSC_Avg_for_one_epoch_Index)
            DSC_Avg_for_one_epoch_Index+=1
            print("DSC standard deviation : ",dsc_std)  
        if DEBUG or DEBUG2:
            break
        print(id(model))
    return model
  
@measure_time
def make_data():
    dataset=eval(f"{DATASET}()")
    train_size=len(dataset)*4//5
    test_size=len(dataset)-train_size
    traindata,valdata = random_split(dataset,[train_size,test_size],generator=torch.Generator().manual_seed(42))
    return traindata,valdata

def make_model():
    model=eval(f"{MODEL_TYPE}(pretrain=True)").to(DEVICE)
    return model

def main():
    traindata,valdata=make_data()
    model=make_model()
    model=pretrain(model,traindata,valdata)
    save_model(model,PRETRAIN_MODEL_PATH)

if __name__=="__main__":
    main()
    #pass