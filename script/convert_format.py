import subprocess
import os
from config import CONFIG



TYPE="nifti"
PREFIX="Segmentation"
CONFIG_PATH=CONFIG["PATH"]
DATASETPATH=CONFIG_PATH["LUNG_RADIOMICS_INTEROBS_PATH"]

#CONVERTER=":/tmp"
#DATASETPATH="/home/s0ran/Strage/CancerSegmentation/dataset/manifest-1603198545583/NSCLC-Radiomics"
#DATASETPATH="/home/share/soran/CancerSegmentation/dataset/manifest-1598890146597/NSCLC-Radiomics-Interobserver1"
#path="/tmp/LUNG1-001/09-18-2008-StudyID-NA-69331/300.000000-Segmentation-9.554"
DOC_DIRNAME="/tmp"

def call(converter,inputdicom,outputdir):
    args=["docker", "run", "-v",converter,"qiicr/dcmqi", "segimage2itkimage", "-t", TYPE,"-p",PREFIX ,"--inputDICOM",inputdicom,"--outputDirectory",outputdir]
    subprocess.run(args)

def main(path):
    converter=f"{path}:{DOC_DIRNAME}"
    for root,dirs,files in os.walk(path):
        dirname=os.path.split(root)[-1]
        if "Segmentation" in dirname:# and ("LUNG1-133" in root or "LUNG1-134" in root or "LUNG1-140" in root or"LUNG1-141" in root or "LUNG1-142" in root or "LUNG1-145" in root or "LUNG1-146" in root or "LUNG1-147" in root or "LUNG1-149" in root or "LUNG1-157" in root or "LUNG1-161" in root or "LUNG1-163" in root or "LUNG1-187" in root or "LUNG1-188" in root or "LUNG1-195" in root or "LUNG1-200" in root ):
            #print(os.path.splitext(files[0]))
            dcmfiles=[s for s in files if os.path.splitext(s)[-1]==".dcm"]
            assert len(dcmfiles)==1, print(dcmfiles)
            filename=os.path.join(root,dcmfiles[0])
            filename=convert_filename(path,filename)
            dirname=convert_filename(path,root)
            #print(f"converter: {converter}\n dcmfile: {filename}\n dir: {dirname}")
            call(converter,filename,dirname)



def convert_filename(path,filename):
    filename=filename.replace(path,DOC_DIRNAME)
    return filename



if __name__=="__main__":
    main(DATASETPATH)




