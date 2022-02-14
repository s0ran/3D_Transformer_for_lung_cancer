import torch 
from torch.optim  import Adam,SGD,RMSprop
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
from loss_function import DiceLoss, DSLManager,BinaryDiceLoss,BinaryDSLManager,BCE_with_DiceLoss,weightDiceLoss,MSELoss
from metrics import DSC, DSCManager,BinaryDSC,BinaryDSCManager,weightDSC,BinaryweightDSC
from utility import OutputLogger, measure_time
from model import VolumeTransformer,ThreeDimensionalUNet,load_model,save_model,MiniVolumeTransformer,VolumeTransformer2,VolumeTransformer3,VolumeTransformer4,VolumeTransformer5,VolumeTransformer6,MiniTransformer,VolumeTransformer7,MiniTransformer2,MiniTransformer3,MiniTransformer4,MiniTransformer5,MiniTransformer8,Transformer2,Transformer3,Transformer4
from config import CONFIG

now=datetime.datetime.now()
CONFIG_PATH=CONFIG["PATH"]
CONFIG_PRETRAIN=CONFIG["Pretraining"]
Datasets={
    0:"VolumeLungRadiomicsInterobserverDataset",
    1:"VolumeLungRadiomicsDataset",
}
Keylist={
    0:"GTV-1vis-5",
    1:"GTV-1",
    }
Models={
    0:"VolumeTransformer",
    1:"VolumeConTransformer",
    2:"ThreeDimensionalUNet",
    3:"MiniVolumeTransformer",
    4:"VolumeTransformer2",
    5:"VolumeTransformer3",
    6:"VolumeTransformer4",
    7:"VolumeTransformer5",
    8:"VolumeTransformer6",
    9:"MiniTransformer",
    10:"VolumeTransformer7",
    11:"MiniTransformer2",
    12:"MiniTransformer3",
    13:"MiniTransformer4",
    14:"MiniTransformer5",
    17:"MiniTransformer8",
    18:"Transformer2",
    19:"Transformer3",
    20:"Transformer4",
}
DATASET_ID=CONFIG_PRETRAIN.getint("DATASET_ID")
DATASET=Datasets[DATASET_ID]
MODEL_TYPE=Models[CONFIG_PRETRAIN.getint("MODEL_ID")]
LABEL_KEY=Keylist[DATASET_ID]
BATCH_SIZE=CONFIG_PRETRAIN.getint("BATCH_SIZE")
DEVICE=torch.device(CONFIG_PRETRAIN.get("DEVICE")) if torch.cuda.is_available() else torch.device("cpu")
DEBUG=False
DEBUG2=False
#USE_PRETRAINMODEL=CONFIG_PRETRAIN.getboolean("USE_PRETRAINMODEL")
#PREPRETRAIN_MODEL_PATH=os.path.join(CONFIG_PATH["PRETRAIN_MODELPATH"],CONFIG_PRETRAIN["PRETRAIN_FILE_NAME"])
ID=f"{str(now.date())}-{now.hour}-{now.minute}"
PRETRAIN_PATH_BASE=CONFIG_PATH["PRETRAIN_MODELPATH"]
PRETRAIN_MODEL_PATH=os.path.join(CONFIG_PATH["PRETRAIN_MODELPATH"],f"{ID}.pt")
#OUTPUTLOG_PATH=os.path.join(CONFIG_PATH["PRETRAIN_LOGPATH"],f"{ID}.txt")
EPOCH=CONFIG_PRETRAIN.getint("EPOCH")
LEARNINGLATE=CONFIG_PRETRAIN.getfloat("LEARNINGLATE")
WRITER=SummaryWriter()
#OGGER=OutputLogger(OUTPUTLOG_PATH) if CONFIG_PRETRAIN.getboolean("OUTPUT_LOG") else None
TQDM_ENABLED=CONFIG_PRETRAIN.getboolean("TQDM_ENABLED")
NUM_WORKERS=CONFIG_PRETRAIN.getint("NUM_WORKERS")

Loss_for_one_patch_Index=0
DSC_on_evaluation_Index=0
DSC_Avg_for_one_epoch_Index=0
DSC_on_training_Index=0

torch.backends.cudnn.benchmark = True
#torch.backends.cudnn.enabled = False

def information():
    print(ID)
    print(MODEL_TYPE)
    print(DATASET)
    print(DEVICE)
    print(LABEL_KEY)

@measure_time
def train(model,traindata,valdata):
    global WRITER,DSC_on_evaluation_Index,DSC_Avg_for_one_epoch_Index,DSC_on_training_Index,Loss_for_one_patch_Index
    loss_function=DiceLoss
    metric=DSC
    optimizer=Adam(model.parameters(),lr=LEARNINGLATE)
    for epoch in tqdm(range(1,EPOCH+1),disable=not TQDM_ENABLED):
        print(f"------------EPOCH {epoch}/{EPOCH}------------")
        model.pre_train()
        trainloader=DataLoader(traindata,batch_size=BATCH_SIZE,num_workers=NUM_WORKERS)
        for data,label in tqdm(trainloader,disable=not TQDM_ENABLED):
            #print(data.size())
            data=data.to(DEVICE)
            label=data.to(DEVICE)
            optimizer.zero_grad()
            pre=model(data)
            loss=loss_function(pre,label)
            WRITER.add_scalar("process/loss_for_one_patch",loss,Loss_for_one_patch_Index)
            Loss_for_one_patch_Index+=1
            #print(loss)
            torch.cuda.empty_cache()
            loss.backward()
            with torch.no_grad():
                dsc=metric(pre,label)
                WRITER.add_scalar("process/DSC_on_training",dsc,DSC_on_training_Index)
                DSC_on_training_Index+=1
                print("Image Dice Similarity Coefficient",dsc)
                print("finish training")
                print(f"whole loss {loss}")
            pre=model(torch.flip(data,[2]))
            loss=loss_function(pre,torch.flip(label,[2]))
            WRITER.add_scalar("process/loss_for_one_patch",loss,Loss_for_one_patch_Index)
            Loss_for_one_patch_Index+=1
            #print(loss)
            torch.cuda.empty_cache()
            loss.backward()
            pre=model(data.transpose(3,4))
            loss=loss_function(pre,label.transpose(3,4))
            WRITER.add_scalar("process/loss_for_one_patch",loss,Loss_for_one_patch_Index)
            Loss_for_one_patch_Index+=1
            #print(loss)
            torch.cuda.empty_cache()
            loss.backward()
            pre=model(torch.flip(data.transpose(3,4),[2]))
            loss=loss_function(pre,torch.flip(label.transpose(3,4),[2]))
            WRITER.add_scalar("process/loss_for_one_patch",loss,Loss_for_one_patch_Index)
            Loss_for_one_patch_Index+=1
            #print(loss)
            torch.cuda.empty_cache()
            loss.backward()
            pre=model(torch.flip(data,[3,4]))
            loss=loss_function(pre,torch.flip(label,[3,4]))
            WRITER.add_scalar("process/loss_for_one_patch",loss,Loss_for_one_patch_Index)
            Loss_for_one_patch_Index+=1
            #print(loss)
            torch.cuda.empty_cache()
            loss.backward()
            pre=model(torch.flip(data.transpose(3,4),[2,3,4]))
            loss=loss_function(pre,torch.flip(label.transpose(3,4),[2,3,4]))
            WRITER.add_scalar("process/loss_for_one_patch",loss,Loss_for_one_patch_Index)
            Loss_for_one_patch_Index+=1
            #print(loss)
            torch.cuda.empty_cache()
            loss.backward()
            pre=model(torch.flip(data,[2,3,4]))
            loss=loss_function(pre,torch.flip(label,[2,3,4]))
            WRITER.add_scalar("process/loss_for_one_patch",loss,Loss_for_one_patch_Index)
            Loss_for_one_patch_Index+=1
            #print(loss)
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            #dsc=metric(pre,label,keys=LABEL_KEYS)
            
            del data,loss,pre,label
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
            for data,label in tqdm(valloader,disable=not TQDM_ENABLED):
                data=data.to(DEVICE)
                label=data.to(DEVICE)
                pre=model(data)
                loss=loss_function(pre,label)
                dsc=metric(pre,label)
                WRITER.add_scalar("process/DSC_on_evaluation",dsc,DSC_on_evaluation_Index)
                DSC_on_evaluation_Index+=1
                #print(dsc,loss)
                loss_list.append(loss)
                dsc_list.append(dsc)
                del data,label,pre
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
        save_model(model,os.path.join(PRETRAIN_PATH_BASE,f"{ID}-{epoch}.pt"))
        if epoch>=2:
            os.remove(os.path.join(PRETRAIN_PATH_BASE,f"{ID}-{epoch-1}.pt"))
    return model
  
@measure_time
def make_data():
    dataset=eval(f"{DATASET}(key=LABEL_KEY)")
    train_size=len(dataset)*4//5
    test_size=len(dataset)-train_size
    traindata,valdata = random_split(dataset,[train_size,test_size],generator=torch.Generator().manual_seed(42))
    return traindata,valdata

def make_model():
    #if USE_PRETRAINMODEL:
    #    model=load_model(PREPRETRAIN_MODEL_PATH,DEVICE,MODEL_TYPE)
    #else:
    model=eval(f"{MODEL_TYPE}(pretrain=True)").to(DEVICE)
    return model

def main():
    information()
    traindata,valdata=make_data()
    model=make_model()
    model=train(model,traindata,valdata)
    save_model(model,PRETRAIN_MODEL_PATH)

if __name__=="__main__":
    main()