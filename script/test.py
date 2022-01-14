import torch

from metrics import DSC_for_list

if __name__=="__main__":
    a=torch.rand(1780,2,512,512)
    b=torch.rand(1780,2,512,512)
    print(a.dtype)
    print(a.size())
    print(DSC_for_list(a,b))