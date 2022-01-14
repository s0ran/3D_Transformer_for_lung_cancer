import os
import nibabel as nib

DATASETPATH="/home/s0ran/Strage/CancerSegmentation/dataset/manifest-1603198545583/NSCLC-Radiomics"

def main():
    for root,dirs,files in os.walk(DATASETPATH):
        if "Segmentation" in os.path.split(root)[-1]:
            filepath=os.path.join(root,"Segmentation-GTV-1.nii.gz")
            assert os.path.isfile(filepath)
            image=nib.load(filepath)
            print(image.shape)
            print(type(image))
            image=image.get_fdata()
            print(image.shape)
            print(image.astype("int16"))
            break
            #assert "Segmentation-GTV-1.nii" in files, f"path{root} does not satisfy the condition \nfiles:{files}"
    print("any problems is not found")
        
if __name__=="__main__":
    main()