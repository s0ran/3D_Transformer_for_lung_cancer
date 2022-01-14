import os
from posixpath import splitext
from re import X
import sys
import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset,DataLoader
from config import CONFIG
import itertools
import math
from tqdm import tqdm
#from sklearn.model_selection import train_test_split
#from scipy.ndimage import zoom
import gc
import nibabel as nib

from utility import measure_time
from memory_profiler import profile

CONFIG_DATASET=CONFIG["Dataset"]
PATCH_SIZE=CONFIG_DATASET.getint("PATCHSIZE")
BLOCK_SIZE=CONFIG_DATASET.getint("BLOCKSIZE")
IMAGE_SIZE=CONFIG_DATASET.getint("IMAGESIZE")

class BaseDataset (Dataset):
    def __init__(self,path,base,label=True,patients=None,patients_path=None):
        #self.raw_data
        #self.segmentation
        self.label=label
        self.base=base
        self.root=path
        self.patients=os.listdir(path) if patients==None else patients
        self.patients_path=os.listdir(path) if patients_path==None else patients_path
        #print(self.patients,self.patients_path)
        #if patients==None or patients_path==None:
        self.check_data()

    def check_data(self):
        patients_list=[]
        patients_path_list=[]
        for idx,patient in tqdm(enumerate(self.patients)):
            path=self.get_path(idx)
            for root,dirs,files in os.walk(path):
                #for file in files:
                #file=files[0]
                if files:
                    file=files[0]
                    with pydicom.dcmread(os.path.join(root,file),force=True) as dc:
                        if dc.Modality=="CT" and len(dc.pixel_array.shape)==2 and not np.any(np.isnan(dc.pixel_array)):
                            try:
                                if not dc.Rows==dc.Columns==512:
                                    print(dc.Row,dc.Columns)
                                    raise ValueError(f"{dc.Row},{dc.Column}")                        
                            except:
                                break                  
                            assert dc.Rows==dc.Columns==512
                            
                            #print(dc.Rows,dc.Columns)
                            patients_list.append(self.patients[idx])
                            patients_path_list.append(root)
                        #return 0
                    break
        self.patients_path=patients_path_list
        self.patients=patients_list

    def __len__(self):
        return len(self.patients)

    def get_path(self,idx):
        return os.path.join(self.root,self.patients_path[idx])

    def get_patient_id(self,dirname):
        return dirname.removeprefix(self.base)

    def __getitem__(self,idx):
        if idx>=len(self.patients):
            raise IndexError
        #for root,dirs,files in os.walk(path):
        patient_id=self.get_patient_id(self.patients[idx])
        patient_path=self.get_path(idx)
        #print(patient_id)
        data=[]
        files=os.listdir(patient_path)
        #print(files)
        files.sort(reverse=True)
        for file in files:
            file_path=os.path.join(patient_path,file)
            with pydicom.dcmread(file_path,force=True) as dc:
                image=dc.pixel_array
                #data.append(torch.tensor(image.astype("int16")))
        data=torch.stack(data) 
        #print(data)
        if self.label:
            patch_provider=LabeledPatchProvider(data)
        else:
            patch_provider=PatchProvider(data)        
        return patient_id,patch_provider

class SegmentationDataset(BaseDataset):
    def __init__(self,path,base,keys=[]):
        self.descriptions=keys
        super().__init__(path,base)
        

    def check_data(self):
        patients_list=[]
        seg_path_list=[]
        patients_path_list=[]
        #print(self.patients)
        for idx,patient in tqdm(enumerate(self.patients)):
            #print(patient)
            path=self.get_path(idx)
            #print(path)
            if os.path.split(path)[-1]=="LICENSE":
                continue
            dic=dict({})
            raw_size=0
            seg_size=0

            for root,dirs,files in os.walk(path):
                #print(root,dirs,files)
                if files:
                    file=files[0]
                    dirname=os.path.split(root)[-1]
                    if "Segmentation" in dirname:
                        filepaths=[]
                        dic["seg"]={}
                        #print(self.descriptions)
                        for key in self.descriptions:
                            filepath=os.path.join(root,f"Segmentation-{key}.nii.gz")
                            filepaths.append(filepath)
                            if os.path.isfile(filepath):
                                dic["seg"][key]=filepath
                                seg_size=nib.load(filepath).shape[2]                 
                    elif len(files)!=1:
                        #print(root)                      
                        with pydicom.dcmread(os.path.join(root,file),force=True) as dc:
                            #print(type(dc.pixel_array))
                            raw_size=len(files)
                            #print(len(files))
                            dic["raw"]=root
            if "raw" in dic.keys() and "seg" in dic.keys() and seg_size==raw_size:
            #(seg_shape[0]%raw_size)==0:
                patients_list.append(self.patients[idx])
                patients_path_list.append(dic["raw"])
                seg_path_list.append(dic["seg"])
                #interpolated_path_list.append(dic["itp"])
            else:
                pass
                #print(root)
        
        self.patients_path=patients_path_list
        self.patients=patients_list
        self.seg_path=seg_path_list

    @measure_time
    def __getitem__(self,idx):
        #print(f"len(self){len(self)} -idx {idx}")
        if idx>=len(self):
            return IndexError
        patient_path=self.patients_path[idx]
        seg_paths=self.seg_path[idx]
        data=[]
        files=os.listdir(patient_path)
        files.sort(reverse=True)
        for file in files:
            file_path=os.path.join(patient_path,file)
            with pydicom.dcmread(file_path,force=True) as dc:
                image=dc.pixel_array
                data.append(torch.tensor(image.astype("int16")))
        data=torch.stack(data) 
        segs={}
        for key,seg_path in seg_paths.items():
            seg=torch.tensor(nib.load(seg_path).get_fdata().astype("int16")) 
            seg=torch.permute(seg,(2,1,0))
            segs[key]=seg
        descriptions=self.descriptions

        #print(data.size())
        #print(seg.size())
        """with pydicom.dcmread(seg_path,force=True) as dc:
            #print(dc.pixel_array.shape)
            seg=torch.tensor(dc.pixel_array.astype("int16"))
            for seq in dc.SegmentSequence:
                #print(dir(seq))
                descriptions.append(seq.SegmentDescription)"""
            #print(seg.dtype)
        
        return LabeledPatchProvider(data,labels=segs,descriptions=descriptions,cut_zeros=False,cut_key=self.descriptions[0])

    def __len__(self):
        return len(self.patients_path)

class PretrainingDataset(BaseDataset):
    def __init__(self,path,base,patients=None,patients_path=None):
        super().__init__(path,base,patients=patients,patients_path=patients_path)
        self.base_idx=0
        self.pos=0
        self.size=self.set_len()       

    def set_len(self):
        size=0
        #print(super().__len__())
        for i in tqdm(range(super().__len__())):
            size+=len(super().__getitem__(i)[1])
            #print(i)
        #print(size)
        return size

    def __len__(self):
        return self.size

    def __getitem__(self,idx):
        if len(super().__getitem__(self.base_idx)[1])<=(idx-self.pos):
            self.pos=idx
            self.base_idx+=1
        return super().__getitem__(self.base_idx)[1][idx-self.pos]

class PatchProvider:
    def __init__(self,cube,patch_size=PATCH_SIZE,block_size=BLOCK_SIZE,image_size=IMAGE_SIZE,labels=None,descriptions=None,cut_zeros=False,cut_key=None):
        self.cube=self.normalize(cube,cube.min(),cube.max())
        self.patch_size=patch_size
        self.block_size=block_size
        self.image_size=image_size
        self.label=None
        self.original_size=self.cube.size()
        self.cut_zeros=cut_zeros    
        self.padding(labels)
        self.id_list=[] if cut_zeros else list(range(len(self))) 
        if cut_zeros:
            self.cut(key=cut_key)

    def padding(self,labels):
        #print(label.size())
        layers=self.block_size-self.cube.size()[0]%self.block_size
        padding_layer=torch.zeros([layers,self.image_size,self.image_size])
        self.cube=torch.cat([self.cube,padding_layer])
        if labels!=None:
            #label=self.normalize(label,label.min(),label.max())
            #print(label.size(),self.original_size)
            #labels=torch.split(label,self.original_size[0])
            self.labels={}
            #print(labels[0].size())
            #print(len(labels),len(descriptions))
            #print(descriptions)
            for key,value in labels.items():
                label=torch.cat([value,padding_layer])
                self.labels[key]=self.normalize(label,label.min(),label.max())
            """for i,label in enumerate(labels):
                if key:
                    if key==descriptions[i]:
                        self.labels[descriptions[i]]=torch.cat([label,padding_layer])
                else:
                    self.labels[descriptions[i]]=torch.cat([label,padding_layer])              
            del labels"""
            gc.collect()
        else:
            self.label=self.cube
            self.labels={"label":self.label}

    def cut(self,key=None):
        size=self.cube.size()
        for idx in range(size[0]//self.block_size*size[1]//self.block_size*size[2]//self.block_size):
            x_size=size[2]//self.block_size
            y_size=size[1]//self.block_size
            x_index,y_index,z_index=self.axis_index(idx,x_size,y_size)
            x_start,x_end,y_start,y_end,z_start,z_end=self.axis_range(x_index,y_index,z_index)
            if key:
                label=self.labels[key]
            else:
                label=list(self.labels.values())[0]
            label_patch=label[z_start:z_end,y_start:y_end,x_start:x_end]
            #print(label_patch.size())
            if label_patch.max()!=0:
                self.id_list.append(idx)

    def __len__(self):
        if self.cut_zeros:
            return len(self.id_list)
        else:
            size=self.cube.size()
            return size[0]//self.block_size*size[1]//self.block_size*size[2]//self.block_size

    def axis_index(self,idx,x_size,y_size):
        x_index=idx%x_size
        y_index=(idx%(x_size*y_size))//x_size
        z_index=idx//(x_size*y_size)
        return x_index,y_index,z_index

    def axis_range(self,x_index,y_index,z_index):
        x_start=self.block_size*x_index
        x_end=x_start+self.block_size
        y_start=self.block_size*y_index
        y_end=y_start+self.block_size
        z_start=self.block_size*z_index
        z_end=z_start+self.block_size
        return x_start,x_end,y_start,y_end,z_start,z_end

    @measure_time
    def custumgetitem(self,idx,with_label=False):
        size=self.cube.size()
        #print(size)
        x_size=size[2]//self.block_size
        y_size=size[1]//self.block_size
        x_index,y_index,z_index=self.axis_index(idx,x_size,y_size)
        x_start,x_end,y_start,y_end,z_start,z_end=self.axis_range(x_index,y_index,z_index)
        data=torch.zeros(self.patch_size*self.block_size,self.patch_size*self.block_size,self.patch_size*self.block_size)
        for i in range(self.patch_size**3):
            x_pos,y_pos,z_pos=self.axis_index(i,self.patch_size,self.patch_size)
            #print(x_pos,y_pos,z_pos)
            patch_x_start,patch_y_start,patch_z_start=(x_start+self.block_size*(x_pos-1),y_start+self.block_size*(y_pos-1),z_start+self.block_size*(z_pos-1))
            patch_x_end,patch_y_end,patch_z_end=(patch_x_start+self.block_size,patch_y_start+self.block_size,patch_z_start+self.block_size)
            #print(i)
            #print(patch_x_start,patch_y_start,patch_z_start)
            #print(patch_x_end,patch_y_end,patch_z_end)

            if x_pos==(self.patch_size//2) and y_pos==(self.patch_size//2) and z_pos==(self.patch_size//2) and list(self.labels.keys())[0]=="label":
                continue
            if patch_x_start>=0 and patch_y_start>=0 and patch_z_start>=0 and patch_x_end<=size[2] and patch_y_end<=size[1] and patch_z_end<=size[0]:
                #print(data.size())
                #print(self.cube[patch_z_start:patch_z_end,patch_y_start:patch_y_end,patch_x_start:patch_x_end].size())
                data[z_pos*self.block_size:(z_pos+1)*self.block_size,y_pos*self.block_size:(y_pos+1)*self.block_size,x_pos*self.block_size:(x_pos+1)*self.block_size]=self.cube[patch_z_start:patch_z_end,patch_y_start:patch_y_end,patch_x_start:patch_x_end]
        #print(x_pos,y_pos,z_pos)
        if with_label:
            #label=self.cube[z_start:z_end,y_start:y_end,x_start:x_end]
            labels={}
            for key,label in self.labels.items():
                #print(z_start,z_end,y_start,y_end,x_start,x_end)
                label_patch=label[z_start:z_end,y_start:y_end,x_start:x_end]
                if key!="label":
                    labels[key]=torch.stack([label_patch,1-label_patch])
                else:
                    labels[key]=torch.unsqueeze(label_patch,dim=0)
            
            #data=torch.stack([data,1-data])
            data=torch.unsqueeze(data,0)
            #label=torch.stack([label,1-label])
            return data,labels
        else:
            max=data.max()
            min=0
            data=self.normalize(data,min,max)
            #data=torch.stack([data,1-data])
            data=torch.unsqueeze(data,0)
            return data
            
    def __getitem__(self,idx):
        if idx>=len(self):
            raise IndexError
        return self.custumgetitem(self.id_list[idx],with_label=False)

    def normalize(self,data,min,max):
        #print(min,max)
        if max==min:
            return torch.zeros_like(data)
        return (data-min)/(max-min)

    def __iter__(self):
        self.i=0
        return self

    def __next__(self):
        i=self.i
        if i<len(self):
            self.i+=1
            return self.__getitem__(i)
        raise StopIteration
    
    @measure_time
    def rebuild(self,patches):
        #patches[0].size()==B,C,H,W,D
        patches=torch.cat(patches,dim=0)
        size=patches.size()
        #print(self.cube.size())
        #print(patches.size())
        height,width,depth=size[2:5]
        x_size=self.image_size//height
        y_size=self.image_size//width
        z_size=self.image_size//depth
        new_patches=[]
        #print(len(patches),x_size)
        for i in range(0,len(patches),x_size):
            #print(i)
            #print(patches[i:i+x_size].size())
            new_patches.append(torch.cat(list(patches[i:i+x_size]),dim=1))
            #print(new_patches[0].size())

        patches=new_patches
        new_patches=[]
        for j in range(0, len(patches),y_size):
            new_patch=torch.cat(patches[j:j+y_size],dim=2)
            #print(new_patch.size())
            new_patches.append(new_patch)
        #print(new_patches[0].size())
        new_patches=torch.cat(new_patches,dim=3)

        return new_patches
        
class LabeledPatchProvider(PatchProvider):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    def __getitem__(self,idx):
        if idx>=len(self):
            raise IndexError
        #print(f"idx{idx}super()idlist[idx]{super().self.id_list[idx]}")
        return super().custumgetitem(self.id_list[idx],with_label=True)

class LungCTPretrainingDataset(PretrainingDataset):
    def __init__(self,patients=None,patients_path=None):
        path="/home/share/soran/CancerSegmentation/dataset/manifest-1608669183333/Lung-PET-CT-Dx"
        super().__init__(path,"Lung_Dx-",patients=patients,patients_path=patients_path)

class LungCTFullImageDataset(BaseDataset):
    def __init__(self,patients=None,patients_path=None):
        path="/home/share/soran/CancerSegmentation/dataset/manifest-1608669183333/Lung-PET-CT-Dx"
        super().__init__(path,"Lung_Dx-",patients=patients,patients_path=patients_path)#,label=False)

    def collate_fn(self,data):
        image,label=zip(*data[0][1])
        image,label=torch.stack(image),torch.stack(label)
        return image,label

class LungRadiomicsInterobserverDataset(SegmentationDataset):
    def __init__(self,keys=["GTV-1vis-5"]):
        path="/home/share/soran/CancerSegmentation/dataset/manifest-1598890146597/NSCLC-Radiomics-Interobserver1"
        super().__init__(path,"interobs",keys=keys)

    def collate_fn(self,data):
        #print(data)
        return data

class LungRadiomicsDataset(SegmentationDataset):
    def __init__(self,keys=["GTV-1"]):
        path="/home/nfs/snakagawa/Strage/CancerSegmentation/dataset/manifest-1603198545583/NSCLC-Radiomics"
        super().__init__(path,"LUNG1-",keys=keys)


if __name__=="__main__":
    print("start")
    dataset=LungRadiomicsDataset()
    print(dataset[0])

    
