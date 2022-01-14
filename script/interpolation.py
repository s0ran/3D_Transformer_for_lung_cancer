import pydicom
import numpy as np 
from scipy.ndimage import zoom
import os
from utility import measure_time
from tqdm import tqdm

BASE_DIR="/home/s0ran/Strage/CancerSegmentation/dataset/manifest-1598890146597/NSCLC-Radiomics-Interobserver1"

@measure_time
def interpolation():
    for root,dirs,files in tqdm(os.walk(BASE_DIR)):
        if len(files)>1:
            images=[]
            for file in files:
                with pydicom.dcmread(os.path.join(root,file)) as dc:
                    image=dc.pixel_array
                    images.append(image)
            path_list=list(os.path.split(root))
            #print(path_list)
            last=path_list.pop()
            last+=".npy"
            path_list.append(last.replace("NA","SP"))
            
            #print(path_list)
            outputname=os.path.join(*path_list)
            images=np.array(images)
            #print(images.shape)
            #print(images.dtype)
            #interpolated=images
            interpolated=zoom(images,(10,1,1))
            np.save(outputname,interpolated)

if __name__=="__main__":
    interpolation()
    print("complete interpolation")