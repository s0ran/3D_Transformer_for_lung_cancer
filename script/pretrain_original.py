import os
import sys
import torch
from torch.utils.data import DataLoader,random_split
from torch.optim import Adam
from tqdm import tqdm
#from sklearn.model_selection import train_test_split
import datetime


from dataset import LungCTPretrainingDataset,LungCTFullImageDataset
from utility import measure_time
from loss_function import DiceLoss, generalDiceLoss
from model import ThreeDimensionalTransformer,save_model,load_model
from config import CONFIG
from metrics import generalDSC

now=datetime.datetime.now()
CONFIG_PATH=CONFIG["PATH"]
MODEL_PATH=os.path.join(CONFIG_PATH["PRETRAIN_MODELPATH"],f"{str(now.date())}-{now.hour}-{now.minute}.pt")
EPOCH=20
DEBUG=False
DEVICE= torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

print("torch version:           ",torch.__version__) # => 1.10.0+cu102-> 1.10.+cu113
print("cuda version in torch:   ",torch.version.cuda) # => 10.2-> 11.3
print("cudnn version in torch:  ",torch.backends.cudnn.version()) # => 7605->8200
print()


@measure_time
def pretrain(traindata,valdata):
    model=ThreeDimensionalTransformer().to(DEVICE)
    loss_function=generalDiceLoss
    dice_similarity=generalDSC
    optimizer=Adam(model.parameters(),lr=0.001)
    print("start training")
    
    for epoch in tqdm(range(1,EPOCH+1)):
        print(f"------------EPOCH {epoch}/{EPOCH}------------")
        model.train()
        for _,patchprovider in tqdm(traindata):
            trainloader=DataLoader(patchprovider,batch_size=5)
            pre_list=[]
            for data,labels in tqdm(trainloader):
                data,labels=data.to(DEVICE),{key:value.to(DEVICE) for key,value in labels.items()}
                optimizer.zero_grad()
                #print(data)
                #print(torch.any(torch.isnan(data)))
                pre=model(data)
                pre_list.append(pre)
                loss=DiceLoss(pre,labels)
                #print(loss)
                loss.backward()
                optimizer.step()
                if DEBUG:
                    break
            
            with torch.no_grad():
                #print(pre_list[0].size())
                if DEBUG:
                    pass
                else:
                    predict=patchprovider.rebuild(pre_list)
                    correct=torch.permute(torch.stack([patchprovider.cube,1-patchprovider.cube]),(0,2,3,1)).to(DEVICE)
                    #print(predict.size())
                    #print(correct.size())
                    predict,correct=torch.unsqueeze(predict,0),torch.unsqueeze(correct,0)
                    loss=loss_function(predict,correct)
                    #print(loss)
            if DEBUG:
                break
            #break
        model.eval()
        with torch.no_grad():
            
            loss_list=[]
            dsc_list=[]
            for _,patchprovider in tqdm(valdata):
                valloader=DataLoader(patchprovider,batch_size=5)
                pre_list=[]
                for data,_ in valloader:
                    data=data.to(DEVICE)
                    pre=model(data)
                    pre_list.append(pre)
                #print(len(pre_list))
                #print(pre_list[-1].size())
                predict=patchprovider.rebuild(pre_list)
                correct=torch.permute(torch.stack([patchprovider.cube,1-patchprovider.cube]),(0,2,3,1)).to(DEVICE)
                loss=loss_function(predict,correct)
                dsc=dice_similarity(predict,correct)
                loss_list.append(loss)
                dsc_list.append(dsc)
                
            loss_list=torch.stack(loss_list)
            dsc_list=torch.stack(dsc_list)
            loss_avg=torch.mean(loss_list)
            loss_std=torch.std(loss_list)
            dsc_avg=torch.mean(dsc_list)
            dsc_std=torch.std(dsc_list)
            print("generalDSC average : ",dsc_avg)
            print("generalDSC standard deviation : ",dsc_std)                            
    return model

@measure_time
def make_data():
    #manager=LungSplitManager()
    dataset=LungCTFullImageDataset()
    train_size=len(dataset)*4//5
    test_size=len(dataset)-train_size
    traindata,valdata = random_split(dataset,[train_size,test_size],generator=torch.Generator().manual_seed(42))
    return traindata,valdata

def main():
    traindata,valdata=make_data()
    model=pretrain(traindata,valdata)
    save_model(model,MODEL_PATH)

if __name__=="__main__":
    main()
