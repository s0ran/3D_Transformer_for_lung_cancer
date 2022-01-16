import torch 
from torch.optim  import Adam
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
import os
import sys
from tqdm import tqdm
import datetime
import numpy as np
import matplotlib.pyplot as plt
from memory_profiler import profile
import gc

from dataset import CONFIG_DATASET,VolumeLungRadiomicsDataset,VolumeLungRadiomicsInterobserverDataset
from loss_function import DiceLoss, DSLManager,BinaryDiceLoss,BinaryDSLManager,BCE_with_DiceLoss
from metrics import DSC, DSCManager,BinaryDSC,BinaryDSCManager
from utility import OutputLogger, measure_time
from model import VolumeTransformer,ThreeDimensionalUNet,load_model,save_model
from config import CONFIG

now=datetime.datetime.now()
CONFIG_PATH=CONFIG["PATH"]
CONFIG_TRAIN=CONFIG["Training"]
Datasets={
    0:"VolumeLungRadiomicsInterobserverDataset",
    1:"VolumeLungRadiomicsDataset",
}
Keyslist={
    0:["GTV-1vis-5"],
    1:["GTV-1"],
    }
Models={
    0:"VolumeTransformer",
    1:"ThreeDimensionalUNet"
}
DATASET_ID=CONFIG_TRAIN.getint("DATASET_ID")
DATASET=Datasets[DATASET_ID]
MODEL_TYPE=Models[CONFIG_TRAIN.getint("MODEL_ID")]
LABEL_KEYS=Keyslist[DATASET_ID]
BATCH_SIZE=CONFIG_TRAIN.getint("BATCH_SIZE")
DEVICE=torch.device(CONFIG_TRAIN.get("DEVICE")) if torch.cuda.is_available() else torch.device("cpu")
DEBUG=False
DEBUG2=False
USE_PRETRAINMODEL=CONFIG_TRAIN.getboolean("USE_PRETRAINMODEL")
PRETRAIN_MODEL_PATH=os.path.join(CONFIG_PATH["PRETRAIN_MODELPATH"],CONFIG_TRAIN["PRETRAIN_FILE_NAME"])
ID=f"{str(now.date())}-{now.hour}-{now.minute}"
TRAIN_PATH_BASE=CONFIG_PATH["TRAIN_MODELPATH"]
TRAIN_MODEL_PATH=os.path.join(CONFIG_PATH["TRAIN_MODELPATH"],f"{ID}.pt")
OUTPUTLOG_PATH=os.path.join(CONFIG_PATH["TRAIN_LOGPATH"],f"{ID}.txt")
EPOCH=CONFIG_TRAIN.getint("EPOCH")
LEARNINGLATE=CONFIG_TRAIN.getfloat("LEARNINGLATE")
WRITER=SummaryWriter()
LOGGER=OutputLogger(OUTPUTLOG_PATH) if CONFIG_TRAIN.getboolean("OUTPUT_LOG") else None
TQDM_ENABLED=CONFIG_TRAIN.getboolean("TQDM_ENABLED")
NUM_WORKERS=CONFIG_TRAIN.getint("NUM_WORKERS")

Loss_for_one_patch_Index=0
DSC_on_evaluation_Index=0
DSC_Avg_for_one_epoch_Index=0
DSC_on_training_Index=0

#torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = False

@measure_time
def train(model,traindata,valdata):
    global WRITER,DSC_on_evaluation_Index,DSC_Avg_for_one_epoch_Index,DSC_on_training_Index
    loss_function=DiceLoss
    metric=DSC
    optimizer=Adam(model.parameters(),lr=LEARNINGLATE)
    for epoch in tqdm(range(1,EPOCH+1),disable=not TQDM_ENABLED):
        print(f"------------EPOCH {epoch}/{EPOCH}------------")
        model.train()
        trainloader=DataLoader(traindata,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
        for data,labels in tqdm(trainloader,disable=not TQDM_ENABLED):
            data=data.to(DEVICE)
            labels={key:value.to(DEVICE) for key,value in labels.items()}
            optimizer.zero_grad()
            pre=model(data)
            loss=loss_function(pre,labels,keys=LABEL_KEYS)
            loss.backward()
            optimizer.step()
            dsc=metric(pre,labels,keys=LABEL_KEYS)
            WRITER.add_scalar("process/DSC_on_training",dsc,DSC_on_training_Index)
            DSC_on_training_Index+=1
            print("Image Dice Similarity Coefficient",dsc)
            print("finish training")
            print(f"whole loss {loss}")
            del data,loss,pre,labels
            torch.cuda.empty_cache()
            gc.collect()
            if DEBUG or DEBUG2:
                break
        model.eval()
        print("evaluation")
        with torch.no_grad():  
            loss_list=[]
            dsc_list=[]
            valloader=DataLoader(valdata,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
            for data,labels in tqdm(valloader,disable=not TQDM_ENABLED):
                data=data.to(DEVICE)
                labels={key:value.to(DEVICE) for key,value in labels.items()}
                pre=model(data)
                loss=loss_function(pre,labels,keys=LABEL_KEYS)
                dsc=metric(pre,labels,keys=LABEL_KEYS)

                WRITER.add_scalar("process/DSC_on_evaluation",dsc,DSC_on_evaluation_Index)
                DSC_on_evaluation_Index+=1
                #print(dsc,loss)
                loss_list.append(loss)
                dsc_list.append(dsc)
                del data,labels,pre
                torch.cuda.empty_cache()
                gc.collect()
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
        save_model(model,os.path.join(TRAIN_PATH_BASE,f"{ID}-{epoch}.txt"))
    return model
  
@measure_time
def make_data():
    #dataset=LungRadiomicsInterobserverDataset()
    dataset=eval(f"{DATASET}()")
    #dataset=LungRadiomicsDataset()
    #print(len(dataset))
    train_size=len(dataset)*4//5
    test_size=len(dataset)-train_size
    traindata,valdata = random_split(dataset,[train_size,test_size],generator=torch.Generator().manual_seed(42))
    return traindata,valdata

def make_model():
    if USE_PRETRAINMODEL:
        model=load_model(PRETRAIN_MODEL_PATH,DEVICE,MODEL_TYPE)
    else:
        model=eval(f"{MODEL_TYPE}()").to(DEVICE)
    return model

def main():
    traindata,valdata=make_data()
    model=make_model()
    model=train(model,traindata,valdata)
    save_model(model,TRAIN_MODEL_PATH)

if __name__=="__main__":
    main()