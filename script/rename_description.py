import os
import json
import gzip
import shutil
#DATASETPATH="/home/s0ran/Strage/CancerSegmentation/dataset/manifest-1603198545583/NSCLC-Radiomics"
DATASETPATH="/home/share/soran/CancerSegmentation/dataset/manifest-1598890146597/NSCLC-Radiomics-Interobserver1"

def main():
    for root,dirs,files in os.walk(DATASETPATH):
        dirname=os.path.split(root)[-1]
        if "Segmentation" in dirname:
            filename="Segmentation-meta.json"
            assert filename in files, f"{filename} not found in {root}"
            """if not filename in files:
                print(f"{filename} not found in {root}")"""
            filepath=os.path.join(root,filename)
            converter=make_rename_dic(filepath)
            
            for file in files:
                filepath=os.path.join(root,file)
                if os.path.splitext(file)[-1]==".dcm":
                    #os.rename(filepath,os.path.join(root,"1-1.dcm"))
                    pass
                else:
                    for key,value in converter.items():
                        if key in file and "GTV-1" not in file:
                            renamed_filepath=os.path.join(root,file.replace(key+".",value+"."))
                            print(filepath,renamed_filepath)
                            
                            try:
                                os.rename(filepath,renamed_filepath)
                            except:
                                pass
                            """
                            if value=="GTV-1":
                                unzipped_filepath=renamed_filepath.replace(".gz","")
                                with gzip.open(renamed_filepath,"rb") as f_in:
                                    with open(unzipped_filepath,"wb") as f_out:
                                        shutil.copyfileobj(f_in,f_out)
                            """
                    if "30" in file:
                        os.remove(filepath)
                        

                        

def make_rename_dic(filepath):
    converter=dict()
    with open(filepath) as f:
        contents=json.load(f)['segmentAttributes']
        #contents=json.load(f)['SegmentDescription']
        for content in contents:
            content=content[0]
            converter[str(content["labelID"])]=content["SegmentDescription"]
    return converter



if __name__=="__main__":
    main()
