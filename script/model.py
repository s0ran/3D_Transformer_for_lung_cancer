from sys import path_importer_cache
from unittest.mock import patch
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer,Block,Attention
from timm.models.layers import DropPath,Mlp
import os
from config import CONFIG
from functools import partial
from memory_profiler import profile
from loss_function import BCE_with_DiceLoss


DEVICE=CONFIG["Training"]["DEVICE"]
CONFIG_DATASET=CONFIG["Dataset"]
CONFIG_MDOEL=CONFIG["MODEL"]
PATCH_SIZE=CONFIG_DATASET.getint("PATCHSIZE")
BLOCK_SIZE=CONFIG_DATASET.getint("BLOCKSIZE")
IMAGE_SIZE=CONFIG_DATASET.getint("IMAGESIZE")
MAX_DEPTH=CONFIG_DATASET.getint("MAX_DEPTH")
INPUT_IMAGE_SIZE=BLOCK_SIZE*PATCH_SIZE
EMBED_DIM=CONFIG_MDOEL.getint("EMBED_DIM")
NUM_HEADS=CONFIG_MDOEL.getint("NUM_HEADS")
DEPTH=CONFIG_MDOEL.getint("DEPTH")

def save_model(model,path):
    torch.save(model.state_dict(),path)
    return True

def load_model(path,device,model_type,pretrain=False):
    if pretrain:
        model=eval(f"{model_type}(pretrain=True)")
    else:
        model=eval(f"{model_type}()")
    model.load_state_dict(torch.load(path,map_location=device))
    return model.to(device)

class MyMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,bias1=True,bias2=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features,bias=bias1)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features,bias=bias2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=INPUT_IMAGE_SIZE, patch_size=BLOCK_SIZE, in_chans=2, embed_dim=2*(BLOCK_SIZE**3), norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_size,img_size,img_size)
        patch_size = (patch_size,patch_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1],img_size[2]//patch_size[2])
        ##print(self.grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size,bias=False)
        #self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNDim
        ##print(x.shape)
        #x = self.norm(x)
        ##print(x.shape)
        return x

class VolumePatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=IMAGE_SIZE,img_depth=MAX_DEPTH, patch_size=BLOCK_SIZE, in_chans=2, embed_dim=BLOCK_SIZE**3, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_depth,img_size,img_size)
        patch_size = (patch_size,patch_size,patch_size)
        ##print(img_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1],img_size[2]//patch_size[2])
        ##print(self.grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        ##print(self.num_patches)
        self.flatten = flatten
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        #self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        ##print(x.shape)
        x = self.proj(x)
        ##print(x.shape)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNDim
        return x

class MyAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if scale:
            self.scale=scale
        else:
            self.scale = dim ** -1

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        ##print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MyAttention3(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if scale:
            self.scale=scale
        else:
            self.scale = dim ** -1

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        ##print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1))
        attn-=attn.min()
        attn/=attn.max()
        attn *= self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
class DownAttention(nn.Module):
    def __init__(self, dim, outdim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        #head_dim = dim // num_heads
        self.out_dim=outdim
        self.scale = dim ** -1

        self.qkv = nn.Linear(dim, outdim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(outdim, outdim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        #print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.out_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.out_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class DownAttention2(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,scale=None):
        super().__init__()
        self.num_heads = num_heads
        #head_dim = dim // num_heads
        self.out_dim=out_dim
        if scale:
            self.scale=scale
        else:
            self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, out_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        #print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.out_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.out_dim)
        x = self.proj(x)
        x = self.proj_drop(x)+v.reshape(B, N, self.out_dim)
        return x

class DownAttention3(nn.Module):
    def __init__(self, dim, out_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,scale=None):
        super().__init__()
        self.num_heads = num_heads
        #head_dim = dim // num_heads
        self.out_dim=out_dim
        if scale:
            self.scale=scale
        else:
            self.scale = dim ** -0.5

        self.qkv = nn.Linear(dim, out_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm=nn.LayerNorm([2560,2560])

    def forward(self, x):
        #print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.out_dim // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = q @ k.transpose(-2, -1)
        attn-=attn.min()
        attn/=attn.max()
        attn*=self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, self.out_dim)
        x = self.proj(x)
        x = self.proj_drop(x)+v.reshape(B, N, self.out_dim)
        return x

class UpAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim*num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3 * num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim*self.num_heads, dim*self.num_heads)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        ##print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C*self.num_heads)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class UpAttention2(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim*num_heads
        if scale:
            self.scale=scale
        else:
            self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3 * num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim*self.num_heads, dim*self.num_heads)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        ##print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        orig=torch.cat([x]*self.num_heads, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C*self.num_heads)+orig
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class UpAttention3(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim*num_heads
        if scale:
            self.scale=scale
        else:
            self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3 * num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim*self.num_heads, dim*self.num_heads)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        ##print(x.shape)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1))
        attn-=attn.min()
        attn/=attn.max()
        attn*=self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        orig=torch.cat([x]*self.num_heads, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C*self.num_heads)+orig
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class LastAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim* num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        #self.proj = nn.Linear(dim, dim)
        #self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q,k,v):
        #print(v.shape)
        _,_,C=q.shape 
        B, N, vC = v.shape
        #qkv = torch.stack([,,],dim=2).reshape(B, C, 3, self.num_heads, N).permute(2, 0, 3, 1, 4)
        q, k, v = self.qkv(q.transpose(1,2)).reshape(B, C, self.num_heads, N).permute(0,2,3,1), self.qkv(k.transpose(1,2)).reshape(B, C, self.num_heads, N).permute(0,2,3,1), self.qkv(v.transpose(1,2)).reshape(B, vC, self.num_heads, N).permute(0,2,3,1)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        #print((attn@v).shape)
        #print(v.shape)
        #print(attn.shape)
        x = attn @ v
        #x = self.proj(x)
        #x = self.proj_drop(x).transpose(-2,-1)
        return x

class MyBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,bias1=True,bias2=True,scale=None,patch_count=None):
        super().__init__()
        if patch_count:
            self.norm1 = norm_layer([patch_count,dim])
        else:
            self.norm1=norm_layer(dim)
        self.attn = MyAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,scale=scale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if patch_count:
            self.norm2 = norm_layer([patch_count,dim])
        else:
            self.norm2=norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, bias1=bias1,bias2=bias2)

    def forward(self, x):
        ##print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        x = self.drop_path(self.norm1(self.attn(x)))+x
        ##print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = self.drop_path(self.norm2(self.mlp(x)))+x
        return x

class MyBlock2(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,bias1=True,bias2=True,scale=None,patch_count=None):
        super().__init__()
        if patch_count:
            self.norm1 = norm_layer([patch_count,dim])
        else:
            self.norm1=norm_layer(dim)
        self.attn = MyAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,scale=scale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if patch_count:
            self.norm2 = norm_layer([patch_count,dim])
        else:
            self.norm2=norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, bias1=bias1,bias2=bias2)

    def forward(self, x):
        ##print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        x = self.drop_path(self.attn(self.norm1(x)))+x
        ##print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = self.drop_path(self.mlp(self.norm2(x)))+x
        return x

class MyBlock3(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,bias1=True,bias2=True,scale=None,patch_count=None):
        super().__init__()
        if patch_count:
            self.norm1 = norm_layer([patch_count,dim])
        else:
            self.norm1=norm_layer(dim)
        self.attn = MyAttention3(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,scale=scale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if patch_count:
            self.norm2 = norm_layer([patch_count,dim])
        else:
            self.norm2=norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, bias1=bias1,bias2=bias2)

    def forward(self, x):
        ##print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        x = self.drop_path(self.attn(self.norm1(x)))+x
        ##print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = self.drop_path(self.mlp(self.norm2(x)))+x
        return x

class UpBlock(nn.Module):
    def __init__(self, dim, hiddendim,outdim,num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,bias1=True,bias2=True,patch_count=None):
        super().__init__()
        if patch_count:
            self.norm1 = norm_layer([patch_count,dim*num_heads])
            self.norm2=norm_layer([patch_count,outdim])
        else:
            self.norm1=norm_layer(dim*num_heads)
            self.norm2=norm_layer(outdim)
        self.attn = UpAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        #mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim*num_heads, hidden_features=hiddendim,out_features=outdim, act_layer=act_layer,bias1=bias1,bias2=bias2)
        self.skip_path=False
        if dim*num_heads==outdim:
            self.skip_path=True
            

    def forward(self, x):
        ##print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        x = self.drop_path(self.norm1(self.attn(x)))
        ##print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        if self.skip_path:
            x = x+self.drop_path(self.norm2(self.mlp(x)))
            return x
        else:
            x = self.drop_path(self.norm2(self.mlp(x)))
            return x

class UpBlock2(nn.Module):
    def __init__(self, dim, hiddendim,outdim,num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,bias1=True,bias2=True,patch_count=None,scale=None):
        super().__init__()
        if patch_count:
            self.norm1 = norm_layer([patch_count,dim*num_heads])
            self.norm2=norm_layer([patch_count,outdim])
        else:
            self.norm1=norm_layer(dim*num_heads)
            self.norm2=norm_layer(outdim)
        self.attn = UpAttention2(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,scale=scale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim*num_heads, hidden_features=hiddendim,out_features=outdim, act_layer=act_layer,bias1=bias1,bias2=bias2)
        self.skip_path=False
        if dim*num_heads==outdim:
            self.skip_path=True
            

    def forward(self, x):
        ##print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        x = self.drop_path(self.norm1(self.attn(x)))
        ##print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        if self.skip_path:
            x = x+self.drop_path(self.norm2(self.mlp(x)))
            return x
        else:
            x = self.drop_path(self.norm2(self.mlp(x)))
            return x

class UpBlock3(nn.Module):
    def __init__(self, dim, hiddendim,outdim,num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,bias1=True,bias2=True,patch_count=None,scale=None):
        super().__init__()
        if patch_count:
            self.norm1 = norm_layer([patch_count,dim*num_heads])
            self.norm2=norm_layer([patch_count,outdim])
        else:
            self.norm1=norm_layer(dim*num_heads)
            self.norm2=norm_layer(outdim)
        self.attn = UpAttention3(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,scale=scale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim*num_heads, hidden_features=hiddendim,out_features=outdim, act_layer=act_layer,bias1=bias1,bias2=bias2)
        self.skip_path=False
        if dim*num_heads==outdim:
            self.skip_path=True
            

    def forward(self, x):
        ##print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        x = self.drop_path(self.norm1(self.attn(x)))
        ##print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        if self.skip_path:
            x = x+self.drop_path(self.norm2(self.mlp(x)))
            return x
        else:
            x = self.drop_path(self.norm2(self.mlp(x)))
            return x

class DownBlock(nn.Module):
    def __init__(self, dim, hiddendim,outdim,num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,bias1=True,bias2=True,patch_count=None):
        super().__init__()
        if patch_count:
            self.norm1 = norm_layer([patch_count,hiddendim])
            self.norm2=norm_layer([patch_count,outdim])
        else:
            self.norm1=norm_layer(hiddendim)
            self.norm2=norm_layer(outdim)
        self.attn = DownAttention(dim, hiddendim,num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=hiddendim, hidden_features=hiddendim,out_features=outdim, act_layer=act_layer,bias1=bias1,bias2=bias2)
        self.skip_path=False
        if hiddendim==outdim:
            self.skip_path=True
            

    def forward(self, x):
        ##print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        x = self.drop_path(self.norm1(self.attn(x)))
        ##print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        if self.skip_path:
            x = x+self.drop_path(self.norm2(self.mlp(x)))
            return x
        else:
            x = self.drop_path(self.norm2(self.mlp(x)))
            return x

class DownBlock2(nn.Module):
    def __init__(self, dim, hiddendim,outdim,num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,bias1=True,bias2=True,patch_count=None,scale=None):
        super().__init__()
        if patch_count:
            self.norm1 = norm_layer([patch_count,hiddendim])
            self.norm2=norm_layer([patch_count,outdim])
        else:
            self.norm1=norm_layer(hiddendim)
            self.norm2 = norm_layer(outdim)

        self.attn = DownAttention2(dim, hiddendim,num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,scale=scale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=hiddendim, hidden_features=hiddendim,out_features=outdim, act_layer=act_layer,bias1=bias1,bias2=bias2)
        self.skip_path=False
        if hiddendim==outdim:
            self.skip_path=True
            

    def forward(self, x):
        ##print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        x = self.drop_path(self.norm1(self.attn(x)))
        ##print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        if self.skip_path:
            x = x+self.drop_path(self.norm2(self.mlp(x)))
            return x
        else:
            x = self.drop_path(self.norm2(self.mlp(x)))
            return x

class DownBlock3(nn.Module):
    def __init__(self, dim, hiddendim,outdim,num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,bias1=True,bias2=True,patch_count=None,scale=None):
        super().__init__()
        if patch_count:
            self.norm1 = norm_layer([patch_count,hiddendim])
            self.norm2=norm_layer([patch_count,outdim])
        else:
            self.norm1=norm_layer(hiddendim)
            self.norm2 = norm_layer(outdim)

        self.attn = DownAttention3(dim, hiddendim,num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,scale=scale)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=hiddendim, hidden_features=hiddendim,out_features=outdim, act_layer=act_layer,bias1=bias1,bias2=bias2)
        self.skip_path=False
        if hiddendim==outdim:
            self.skip_path=True
            

    def forward(self, x):
        ##print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        x = self.drop_path(self.norm1(self.attn(x)))
        ##print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        if self.skip_path:
            x = x+self.drop_path(self.norm2(self.mlp(x)))
            return x
        else:
            x = self.drop_path(self.norm2(self.mlp(x)))
            return x

class VolumeBlock(nn.Module):
    def __init__(self, dim, hiddendim,outdim,num_heads, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm,bias1=True,bias2=True,patch_count=None):
        super().__init__()
        if patch_count:
            self.norm1 = norm_layer([patch_count,dim])
            self.norm2 = norm_layer([patch_count,outdim])
        else:
            self.norm1=norm_layer(dim)
            self.norm2 = norm_layer(outdim)
        self.attn = MyAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MyMlp(in_features=dim, hidden_features=hiddendim,out_features=outdim, act_layer=act_layer,bias1=bias1,bias2=bias2)
        self.skip_path=False
        if dim==outdim:
            self.skip_path=True
            

    def forward(self, x):
        ##print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        ##print(x.shape)
        x = self.drop_path(self.norm1(self.attn(x)+x))
        ##print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        if self.skip_path:
            x = x+self.drop_path(self.norm2(self.mlp(x)))
            return x
        else:
            x = self.drop_path(self.norm2(self.mlp(x)))
            return x
        return x

class ThreeDimensionalTransformer(VisionTransformer):
    
    def __init__(self, img_size=INPUT_IMAGE_SIZE, patch_size=BLOCK_SIZE, in_chans=1, num_classes=2, embed_dim=EMBED_DIM, depth=DEPTH,
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        self.num_classes=num_classes
        self.depth=depth
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            MyBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.representation=nn.Linear(self.patch_embed.num_patches,2)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        ##print(x.shape)->B N ED
        x=self.representation(x.transpose(1,2))
        B,_,_=x.shape
        output_shape=(B,self.num_classes)+self.patch_embed.patch_size
        x=torch.reshape(x,output_shape)
        x=self.softmax(x)
        return x

class ThreeDimensionalUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, width_multiplier=1, trilinear=False, use_ds_conv=False,pretrain=False):
        super(ThreeDimensionalUNet, self).__init__()
        #_channels = (32, 64, 128, 256, 512)
        _channels = (2, 4, 8, 16, 32)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c*width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        factor = 2 if trilinear else 1
        self.down3 = Down(self.channels[2], self.channels[3]//factor, conv_type=self.convtype)
        self.down4 = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.up1 = Up(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2 = Up(self.channels[3], self.channels[2] // factor, trilinear)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, trilinear)
        self.up4 = Up(self.channels[1], self.channels[0], trilinear,pad=1)
        self.outc = OutConv(self.channels[0], n_classes)
        self.softmax=nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        #print(x1.shape)
        x2 = self.down1(x1)
        #print(x2.shape)
        x3 = self.down2(x2)
        #print(x3.shape)
        x4 = self.down3(x3)
        #print(x4.shape)
        x5 = self.down4(x4)
        #print(x5.shape)
        x = self.up1(x5, x4)
        #print(x.shape)
        x = self.up2(x4, x3)
        #print(x.shape)
        x = self.up3(x, x2)
        #print(x.shape)
        x = self.up4(x, x1)
        #print(x.shape)
        logits = self.outc(x)
        #print(logits.shape)
        return self.softmax(logits)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None,pad=1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=pad),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=pad),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True,pad=0):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2,pad=0)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,pad=pad)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        ##print(x1.size())
        ##print(x2.size())
        ##print([diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2,diffZ // 2, diffZ - diffZ // 2,])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2,]  
                        )
        
        #print(x1.shape,x2.shape)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        #print(x.shape)

        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        #print(x.shape)
        return self.conv(x)

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class VolumeTransformer(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, embed_dim=2048, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.embed_dim=embed_dim
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_classes=num_classes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=VolumeBlock(dim=2048,hiddendim=1024,outdim=1024,num_heads=2)
        self.block2=VolumeBlock(dim=1024,hiddendim=512,outdim=512,num_heads=4)
        self.block3=VolumeBlock(dim=512,hiddendim=256,outdim=256,num_heads=8)
        self.block4=UpBlock(dim=256,hiddendim=512,outdim=512,num_heads=4)
        self.block5=UpBlock(dim=512,hiddendim=1024,outdim=1024,num_heads=2)
        self.block6=UpBlock(dim=1024,hiddendim=4096,outdim=4096,num_heads=4)
        #self.block7=VolumeBlock(dim=4096,hiddendim=8192,outdim=65536,num_heads=num_heads)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(4096)
        #self.representation=nn.Linear(self.patch_embed.num_patches,1)
        #self.act_layer=act_layer()
        #self.output=nn.Linear(embed_dim,patch_size**3*num_classes)
        self.invconv=nn.ConvTranspose3d(4096,2,kernel_size=self.patch_embed.patch_size,stride=self.patch_embed.patch_size)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        ##print(x.shape)
        x = self.patch_embed(x)
        ##print(x.shape)
        x = self.pos_drop(x + self.pos_embed)
        #x = self.blocks(x)
        x1=self.block1(x)
        x2=self.block2(x1)
        x=self.block3(x2)
        x=self.block4(x)+x2
        x=self.block5(x)+x1
        x=self.block6(x)
        #x=self.block7(x)
        x = self.norm(x)
        #x=self.representation(x.transpose(1,2))
        B,_,_=x.shape
        embed_shape=(B,self.num_classes*self.embed_dim)+self.patch_embed.grid_size
        x=torch.reshape(x,embed_shape)
        x=self.invconv(x)
        x=self.softmax(x)
        return x

class VolumeConTransformer(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2,depth=DEPTH, embed_dim=EMBED_DIM, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.embed_dim=embed_dim
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        #self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_classes=num_classes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.depth=depth
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            MyBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth-1)])
        self.lastblock=VolumeBlock(dim=self.embed_dim,hiddendim=self.embed_dim*2,outdim=self.embed_dim*2,num_heads=1,drop_path=dpr[depth-1], norm_layer=norm_layer, act_layer=act_layer)

        self.norm=partial(nn.LayerNorm, eps=1e-6)(self.embed_dim*self.num_classes)
        #self.representation=nn.Linear(self.patch_embed.num_patches,1)
        #self.act_layer=act_layer()
        #self.output=nn.Linear(embed_dim,patch_size**3*num_classes)
        self.invconv=nn.ConvTranspose3d(self.embed_dim*self.num_classes,self.num_classes,kernel_size=self.patch_embed.patch_size,stride=self.patch_embed.patch_size)
        #self.act_layer=act_layer()
        #self.output=nn.Linear(embed_dim,patch_size**3*num_classes)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x=self.lastblock(x)
        x = self.norm(x)
        B,_,_=x.shape
        embed_shape=(B,self.num_classes*self.embed_dim)+self.patch_embed.grid_size
        x=torch.reshape(x,embed_shape)
        ##print(x.shape)
        x=self.invconv(x)
        x=self.softmax(x)
        return x

class ConFreeTransformer(nn.Module):
    def __init__(self,patch_size=BLOCK_SIZE):
        super().__init__()
        self.patchtransformer=ThreeDimensionalTransformer()
        self.step=patch_size

    def forward(self, x):
        pre=torch.zeros_like(x)
        pre=torch.cat([pre,pre],dim=1)
        maxi,maxj,maxk=x.size()[-3],x.size()[-2],x.size()[-1]
        for i in range(0,maxi,self.step):
            for j in range(0,maxj,self.step):
                for k in range(0,maxk,self.step):
                    lasti,lastj,lastk=min(i+self.step,maxi),min(j+self.step,maxj),min(k+self.step,maxk)
                    padi,padj,padk=(),(),()
                    rangei,rangej,rangek=slice(0,0),slice(0,0),slice(0,0)
                    if i==0:
                        padi=(self.step,0)
                        rangei=slice(i,i+self.step*2)
                    elif lasti==maxi:
                        padi=(0,self.step*2+i-maxi)
                        rangei=slice(i-self.step,lasti)
                    else:
                        padi=(0,0)
                        rangei=slice(i-self.step,i+self.step*2)
                    if j==0:
                        padj=(self.step,0)
                        rangej=slice(j,j+self.step*2)
                    elif lastj==maxj:
                        padj=(0,self.step*2+j-maxj)
                        rangej=slice(j-self.step,lastj)
                    else:
                        padj=(0,0)
                        rangej=slice(j-self.step,j+self.step*2)
                    if k==0:
                        padk=(self.step,0)
                        rangek=slice(k,k+self.step*2)
                    elif lastk==maxk:
                        padk=(0,self.step*2+k-maxk)
                        rangek=slice(k-self.step,lastk)
                    else:
                        padk=(0,0)
                        rangek=slice(k-self.step,k+self.step*2)
                    pad=padk+padj+padi+(0,0,0,0)
                    ##print(pad)
                    input=F.pad(x[:,:,rangei,rangej,rangek],pad)
                    ##print(input.size())
                    pre[:,:,i:lasti,j:lastj,k:lastk]=self.patchtransformer(input)
                    del input
                    torch.cuda.empty_cache()
        return pre

class ConFreeUnet(nn.Module):
    def __init__(self,patch_size=BLOCK_SIZE):
        super().__init__()
        self.patchtransformer=ThreeDimensionalUNet()
        self.step=patch_size

    def forward(self, x):
        pre=torch.zeros_like(x)
        pre=torch.cat([pre,pre],dim=1)
        maxi,maxj,maxk=x.size()[-3],x.size()[-2],x.size()[-1]
        for i in range(0,maxi,self.step):
            for j in range(0,maxj,self.step):
                for k in range(0,maxk,self.step):
                    lasti,lastj,lastk=min(i+self.step,maxi),min(j+self.step,maxj),min(k+self.step,maxk)
                    padi,padj,padk=(),(),()
                    rangei,rangej,rangek=slice(0,0),slice(0,0),slice(0,0)
                    if i==0:
                        padi=(self.step,0)
                        rangei=slice(i,i+self.step*2)
                    elif lasti==maxi:
                        padi=(0,self.step*2+i-maxi)
                        rangei=slice(i-self.step,lasti)
                    else:
                        padi=(0,0)
                        rangei=slice(i-self.step,i+self.step*2)
                    if j==0:
                        padj=(self.step,0)
                        rangej=slice(j,j+self.step*2)
                    elif lastj==maxj:
                        padj=(0,self.step*2+j-maxj)
                        rangej=slice(j-self.step,lastj)
                    else:
                        padj=(0,0)
                        rangej=slice(j-self.step,j+self.step*2)
                    if k==0:
                        padk=(self.step,0)
                        rangek=slice(k,k+self.step*2)
                    elif lastk==maxk:
                        padk=(0,self.step*2+k-maxk)
                        rangek=slice(k-self.step,lastk)
                    else:
                        padk=(0,0)
                        rangek=slice(k-self.step,k+self.step*2)
                    pad=padk+padj+padi+(0,0,0,0)
                    ##print(pad)
                    input=F.pad(x[:,:,rangei,rangej,rangek],pad)
                    ##print(input.size())
                    pre[:,:,i:lasti,j:lastj,k:lastk]=self.patchtransformer(input)
                    del input
                    torch.cuda.empty_cache()
        return pre

class TwoDimensionalTransformer(VisionTransformer):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=16, in_chans=1, num_classes=2, embed_dim=512, depth=DEPTH,
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads)
        self.num_classes=num_classes
        self.depth=depth
        self.embed_dim=embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            MyBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.representation=nn.Linear(self.embed_dim,self.num_classes*patch_size**2)
        self.invconv=nn.ConvTranspose2d(self.embed_dim,self.num_classes,kernel_size=self.patch_embed.patch_size,stride=self.patch_embed.patch_size)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)
        

    def forward(self, x):
        #x = self.forward_features(x)
        x = self.patch_embed(x)
        ##print('Allocated:', round(torch.cuda.memory_allocated(device=DEVICE)/1024**3,1), 'GB')
        ##print('Cached:   ', round(torch.cuda.memory_reserved(device=DEVICE)/1024**3,1), 'GB') 
        ##print(x.shape)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        ##print(x.shape)#->B N ED
        x=self.representation(x)
        ##print(x.shape)#->B N P**2*class
        B,_,_=x.shape
        mid_shape=(B,self.embed_dim)+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),mid_shape)
        x=self.invconv(x)
        ##print(x.shape)
        #x=torch.reshape(x,output_shape)
        x=self.softmax(x)
        return x

class ConvTransformer(nn.Module):
    def __init__(self,n_channels=1,n_classes=2):
        super(ConvTransformer, self).__init__()
        #_channels = (32, 64, 128, 256, 512)
        _channels = (2, 4, 8, 16, 64,128)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c) for c in _channels]

        self.convtype = nn.Conv3d

        self.inc = DoubleConv(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)
        self.down4 = Down(self.channels[3], self.channels[4], conv_type=self.convtype)
        self.down5 = Down(self.channels[4], self.channels[5], conv_type=self.convtype)
        self.up1=UpBlock(dim=self.channels[5],hiddendim=512,outdim=512,num_heads=4)
        self.up2=UpBlock(dim=512,hiddendim=4096,outdim=4096,num_heads=8)
        self.up3=UpBlock(dim=4096,hiddendim=32768,outdim=65536,num_heads=8)

        

    def forward(self,x):
        B,C,D,H,W=x.shape
        x1 = self.inc(x)
        #print(x1.shape)
        x2 = self.down1(x1)
        #print(x2.shape)
        x3 = self.down2(x2)
        #print(x3.shape)
        x4 = self.down3(x3)
        #print(x4.shape)
        x5 = self.down4(x4)
        #print(x5.shape)
        x6 = self.down5(x5) 
        x6=torch.flatten(x6,start_dim=2)
        x6=x6.transpose(1,2)
        #print(x6.shape)
        x=self.up1(x6)
        x=self.up2(x)
        x=self.up3(x)
        x=torch.reshape(x,(B,2,D,H,W))
        return x

class MiniVolumeTransformer(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_classes=num_classes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=1024,outdim=512,num_heads=4)
        self.block2=DownBlock(dim=512,hiddendim=128,outdim=64,num_heads=8)
        self.block5=UpBlock(dim=64,hiddendim=512,outdim=512,num_heads=8)
        self.block6=UpBlock(dim=512,hiddendim=2048,outdim=2048,num_heads=4)
        #self.block7=VolumeBlock(dim=4096,hiddendim=8192,outdim=65536,num_heads=num_heads)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(2048)
        #self.representation=nn.Linear(self.patch_embed.num_patches,1)
        #self.act_layer=act_layer()
        #self.output=nn.Linear(embed_dim,patch_size**3*num_classes)
        self.invconv=nn.ConvTranspose3d(2048,2,kernel_size=self.patch_embed.patch_size,stride=self.patch_embed.patch_size)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        x = self.patch_embed(x)
        #print(x.shape)
        x = self.pos_drop(x + self.pos_embed)
        #x = self.blocks(x)
        x1=self.block1(x)
        #print(x1.shape)
        x=self.block2(x1)
        #print(x.shape)
        x=self.block5(x)+x1
        #print(x.shape)
        x=self.block6(x)
        #print(x.shape)
        #x=self.block7(x)
        x = self.norm(x)
        #x=self.representation(x.transpose(1,2))
        B,_,_=x.shape
        embed_shape=(B,self.num_classes*1024)+self.patch_embed.grid_size
        x=torch.reshape(x,embed_shape)
        x=self.invconv(x)
        x=self.softmax(x)
        return x

class VolumeTransformer2(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3//2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_classes=num_classes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=8192,outdim=1024,num_heads=4)
        self.block2=DownBlock(dim=1024,hiddendim=512,outdim=512,num_heads=8)
        self.block3=DownBlock(dim=512,hiddendim=128,outdim=128,num_heads=8)
        self.block4=UpBlock(dim=128,hiddendim=512,outdim=512,num_heads=4)
        self.block5=VolumeBlock(dim=512,hiddendim=512,outdim=1024,num_heads=4)
        self.block6=UpBlock(dim=1024,hiddendim=4096,outdim=8192,num_heads=4)
        #self.block7=VolumeBlock(dim=4096,hiddendim=8192,outdim=65536,num_heads=num_heads)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(8192)
        #self.representation=nn.Linear(self.patch_embed.num_patches,1)
        #self.act_layer=act_layer()
        #self.output=nn.Linear(embed_dim,patch_size**3*num_classes)
        self.invconv=nn.ConvTranspose3d(8192,2,kernel_size=self.patch_embed.patch_size,stride=self.patch_embed.patch_size)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        x = self.patch_embed(x)
        #print(x.shape)
        x = self.pos_drop(x + self.pos_embed)
        #x = self.blocks(x)
        x1=self.block1(x)
        #print(x1.shape)
        x2=self.block2(x1)
        x3=self.block3(x2)
        x=self.block4(x3)+x2
        #print(x.shape)
        x=self.block5(x)+x1
        #print(x.shape)
        x=self.block6(x)
        #print(x.shape)
        #x=self.block7(x)
        x = self.norm(x)
        #x=self.representation(x.transpose(1,2))
        B,_,_=x.shape
        embed_shape=(B,self.num_classes*4096)+self.patch_embed.grid_size
        x=torch.reshape(x,embed_shape)
        x=self.invconv(x)
        x=self.softmax(x)
        return x

class VolumeTransformer3(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3//2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_classes=num_classes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=8192,outdim=1024,num_heads=4)
        self.block2=DownBlock(dim=1024,hiddendim=512,outdim=512,num_heads=8)
        #self.block3=DownBlock(dim=512,hiddendim=128,outdim=128,num_heads=8)
        self.block3=MyBlock(dim=512,num_heads=8)
        #self.block4=UpBlock(dim=128,hiddendim=512,outdim=512,num_heads=4)
        self.block4=MyBlock(dim=512,num_heads=8)
        self.block5=VolumeBlock(dim=512,hiddendim=512,outdim=1024,num_heads=4)
        self.block6=UpBlock(dim=1024,hiddendim=4096,outdim=8192,num_heads=4)
        #self.block7=VolumeBlock(dim=4096,hiddendim=8192,outdim=65536,num_heads=num_heads)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(8192)
        #self.representation=nn.Linear(self.patch_embed.num_patches,1)
        #self.act_layer=act_layer()
        #self.output=nn.Linear(embed_dim,patch_size**3*num_classes)
        self.invconv=nn.ConvTranspose3d(8192,2,kernel_size=self.patch_embed.patch_size,stride=self.patch_embed.patch_size)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        x = self.patch_embed(x)
        #print(x.shape)
        x = self.pos_drop(x + self.pos_embed)
        #x = self.blocks(x)
        x1=self.block1(x)
        #print(x1.shape)
        x2=self.block2(x1)
        x3=self.block3(x2)
        x=self.block4(x3)+x2
        #print(x.shape)
        x=self.block5(x)+x1
        #print(x.shape)
        x=self.block6(x)
        #print(x.shape)
        #x=self.block7(x)
        x = self.norm(x)
        #x=self.representation(x.transpose(1,2))
        B,_,_=x.shape
        embed_shape=(B,self.num_classes*4096)+self.patch_embed.grid_size
        x=torch.reshape(x,embed_shape)
        x=self.invconv(x)
        x=self.softmax(x)
        return x

class SimpleTransformer(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3//2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_classes=num_classes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=MyBlock(dim=embed_dim,num_heads=2)
        self.block2=MyBlock(dim=embed_dim,num_heads=2)
        self.block3=MyBlock(dim=embed_dim,num_heads=2)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(embed_dim)
        self.invconv=nn.ConvTranspose3d(embed_dim,2,kernel_size=self.patch_embed.patch_size,stride=self.patch_embed.patch_size)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        x = self.patch_embed(x)
        #print(x.shape)
        x = self.pos_drop(x + self.pos_embed)
        #x = self.blocks(x)
        x1=self.block1(x)
        #print(x1.shape)
        x2=self.block2(x1)
        x3=self.block3(x2)
        #x=self.block4(x3)+x2
        #print(x.shape)
        #x=self.block5(x)+x1
        #print(x.shape)
        #x=self.block6(x)
        #print(x.shape)
        ##x=self.block7(x)
        x = self.norm(x3)
        #x=self.representation(x.transpose(1,2))
        B,_,_=x.shape
        embed_shape=(B,self.embed_dim)+self.patch_embed.grid_size
        print(x.shape)
        x=torch.reshape(x,embed_shape)
        x=self.invconv(x)
        x=self.softmax(x)
        return x

class VolumeTransformer4(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #num_patches = self.patch_embed.num_patches

        #self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=8192,outdim=1024,num_heads=4)
        self.block2=DownBlock(dim=1024,hiddendim=512,outdim=512,num_heads=8)
        #self.block3=DownBlock(dim=512,hiddendim=128,outdim=128,num_heads=8)
        self.block3=MyBlock(dim=512,num_heads=8)
        #self.block4=UpBlock(dim=128,hiddendim=512,outdim=512,num_heads=4)
        #self.block4=MyBlock(dim=1024,num_heads=8)
        #self.block5=VolumeBlock(dim=512,hiddendim=512,outdim=1024,num_heads=4)
        self.block5=VolumeBlock(dim=512,hiddendim=512,outdim=1024,num_heads=4)
        self.block6=UpBlock(dim=1024,hiddendim=8192,outdim=65536,num_heads=8)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(65536)
        #self.representation=nn.Linear(self.patch_embed.num_patches,1)
        #self.act_layer=act_layer()
        #self.output=nn.Linear(embed_dim,patch_size**3*num_classes)
        #self.invconv=nn.ConvTranspose3d(8192,2,kernel_size=self.patch_embed.patch_size,stride=self.patch_embed.patch_size)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x = self.patch_embed(x)
        #print(x.shape)
        #x = self.pos_drop(x + self.pos_embed)
        #x = self.blocks(x)
        x1=self.block1(x)
        #print(x1.shape)
        x2=self.block2(x1)
        x=self.block3(x2)+x2
        #x=self.block4(x)
        #print(x.shape)
        x=self.block5(x)+x1
        #print(x.shape)
        x=self.block6(x)
        #print(x.shape)
        #x=self.block7(x)
        #x = self.norm(x)
        #x=self.representation(x.transpose(1,2))
        
        out_shape=(B,self.num_classes,D,H,W)
        #embed_shape=(B,self.num_classes*self.embed_dim)+self.patch_embed.grid_size
        x=torch.reshape(x,out_shape)
        #x=self.invconv(x)
        x=self.softmax(x)
        return x

class VolumeTransformer5(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #num_patches = self.patch_embed.num_patches

        #self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=4096,outdim=1024,num_heads=4)
        self.block2=DownBlock(dim=1024,hiddendim=512,outdim=512,num_heads=8)
        self.block3=MyBlock(dim=512,num_heads=8)
        self.block5=VolumeBlock(dim=512,hiddendim=512,outdim=1024,num_heads=4)
        #self.block6=UpBlock(dim=1024,hiddendim=8192,outdim=65536,num_heads=8)
        self.mlp = Mlp(in_features=1024, hidden_features=8192,out_features=2*self.embed_dim, act_layer=act_layer)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(2*self.embed_dim)
        #self.representation=nn.Linear(self.patch_embed.num_patches,1)
        #self.act_layer=act_layer()
        #self.output=nn.Linear(embed_dim,patch_size**3*num_classes)
        #self.invconv=nn.ConvTranspose3d(2*self.embed_dim,2,kernel_size=self.patch_embed.patch_size,stride=self.patch_embed.patch_size)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x = self.patch_embed(x)
        #print(x.shape)
        #x = self.pos_drop(x + self.pos_embed)
        #x = self.blocks(x)
        x1=self.block1(x)
        #print(x1.shape)
        x2=self.block2(x1)
        x=self.block3(x2)+x2
        #x=self.block4(x)
        #print(x.shape)
        x=self.block5(x)+x1
        #print(x.shape)
        x=self.mlp(x)
        #print(x.shape)
        #x=self.block7(x)
        x = self.norm(x)
        #x=self.representation(x.transpose(1,2))
        
        out_shape=(B,self.num_classes,D,H,W)
        embed_shape=(B,self.num_classes)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        x=self.softmax(x)
        return x

#これ
class VolumeTransformer6(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #num_patches = self.patch_embed.num_patches

        #self.pos_drop = nn.Dropout(p=drop_rate)

        self.num_classes=num_classes
        patch_count=320*512*512//embed_dim
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock3(dim=embed_dim,hiddendim=512,outdim=128,num_heads=4,patch_count=patch_count)
        #self.block2=MyBlock2(dim=128,num_heads=8,patch_count=patch_count,scale=10**-4)
        self.block3=MyBlock3(dim=128,num_heads=8,patch_count=patch_count)
        #self.block4=MyBlock2(dim=128,num_heads=8,patch_count=patch_count,scale=10**-4)
        self.block5=UpBlock3(dim=128,hiddendim=1024,outdim=embed_dim,num_heads=4,patch_count=patch_count)
        self.norm1=partial(nn.LayerNorm, eps=1e-6)([patch_count,embed_dim])
        self.norm2=partial(nn.LayerNorm, eps=1e-6)([patch_count,128])
        #self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)
        self.sigmoid=nn.Sigmoid()
        self.pretrain=pretrain

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x = self.norm1(self.patch_embed(x))
        x1=self.block1(x)
        #x2=self.block2(x1)
        x=self.block3(x1)+x1
        #x=self.norm2(self.block4(x)+x1)
        x=self.block5(x)
        
        out_shape=(B,1,D,H,W)
        embed_shape=(B,1)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        #x=self.softmax(x)
        x=self.sigmoid(x)
        if not self.pretrain:
            x=torch.cat([x,1-x],dim=1)
        return x
    
    def pre_train(self,*args,**kwargs):
        super().train(*args,**kwargs)
        self.pretrain=True

    def train(self,*args,**kwargs):
        if args:
            if args[0]:
                self.pretrain=False
        elif self.pretrain:
            self.pretrain=False
        super().train(*args,**kwargs)

class VolumeTransformer7(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=1024,outdim=1024,num_heads=4)
        self.mha1=nn.MultiheadAttention(1024,num_heads=8) 
        self.mha2=nn.MultiheadAttention(256,num_heads=8) 
        self.mha3=nn.MultiheadAttention(256,num_heads=8) 
        self.block2=DownBlock(dim=1024,hiddendim=256,outdim=256,num_heads=8)
        #DownBlock(dim=2048,hiddendim=32,outdim=32,num_heads=8)
        self.block3=MyBlock(dim=256,num_heads=8)
        self.block4=MyBlock(dim=256,num_heads=8)
        self.block5=UpBlock(dim=256,hiddendim=1024,outdim=1024,num_heads=4)
        self.block6=UpBlock(dim=1024,hiddendim=8192,outdim=embed_dim*num_classes,num_heads=4)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(embed_dim*num_classes)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x0 = self.patch_embed(x)
        x1=self.block1(x0)
        x2=self.block2(x1)
        x3,_=self.mha3(x2,x2,self.block3(x2))
        x,_=self.mha2(x2,x2,self.block4(x3+x2))
        x,_=self.mha1(x1,x1,self.block5(x+x2))
        x=self.block6(x+x1)
        x = self.norm(x)
        #x=self.mha(x0,x0,x)
        out_shape=(B,self.num_classes,D,H,W)
        embed_shape=(B,self.num_classes)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        x=self.softmax(x)
        return x

class MiniTransformer(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=256,outdim=128,num_heads=4)
        
        self.block2=MyBlock(dim=128,num_heads=8)
        #DownBlock(dim=2048,hiddendim=32,outdim=32,num_heads=8)
        self.block3=MyBlock(dim=128,num_heads=8)
        self.mha1=nn.MultiheadAttention(128,num_heads=8)
        self.mha2=nn.MultiheadAttention(128,num_heads=8)
        self.block4=MyBlock(dim=128,num_heads=8)
        self.block5=UpBlock(dim=128,hiddendim=4096,outdim=embed_dim*num_classes,num_heads=4)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(embed_dim*num_classes)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x0 = self.patch_embed(x)
        x1=self.block1(x0)
        x2=self.block2(x1)
        x3,_=self.mha1(x2,x2,self.block3(x2))
        #x=self.block4(x)+x1
        x,_=self.mha1(x1,x1,self.block4(x3))
        x=self.block5(x+x1)
        x = self.norm(x)
        #x=self.mha(x0,x0,x)
        out_shape=(B,self.num_classes,D,H,W)
        embed_shape=(B,self.num_classes)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        x=self.softmax(x)
        return x

class MiniTransformer2(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3//16
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=256,outdim=128,num_heads=4)
        
        self.block2=MyBlock(dim=128,num_heads=8)
        #DownBlock(dim=2048,hiddendim=32,outdim=32,num_heads=8)
        self.block3=MyBlock(dim=128,num_heads=8)
        self.block4=MyBlock(dim=128,num_heads=8)
        self.block5=UpBlock(dim=128,hiddendim=1024,outdim=2048,num_heads=8)
        self.block6=UpBlock(dim=2048,hiddendim=2048*8,outdim=32768*2,num_heads=8)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(32768*2)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x0 = self.patch_embed(x)
        x1=self.block1(x0)
        x2=self.block2(x1)
        x3=self.block3(x2)+x2
        x=self.block4(x3)+x1
        x=self.block5(x)+x0
        x=self.block6(x)
        x = self.norm(x)
        #x=self.mha(x0,x0,x)
        out_shape=(B,self.num_classes,D,H,W)
        embed_shape=(B,self.num_classes)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        x=self.softmax(x)
        return x

class MiniTransformer3(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3//16
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=512,outdim=16,num_heads=8)
        
        self.block2=MyBlock(dim=16,num_heads=8)
        #DownBlock(dim=2048,hiddendim=32,outdim=32,num_heads=8)
        #self.block3=MyBlock(dim=128,num_heads=8)
        #self.block4=MyBlock(dim=128,num_heads=8)
        #self.block5=UpBlock(dim=128,hiddendim=1024,outdim=2048,num_heads=8)
        self.block6=UpBlock(dim=16,hiddendim=16*8,outdim=32768*2,num_heads=8)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(32768*2)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x0 = self.patch_embed(x)
        x1=self.block1(x0)
        x=self.block2(x1)+x1
        #x3=self.block3(x2)+x2
        #x=self.block4(x3)+x1
        #x=self.block5(x)+x0
        x=self.block6(x)
        x = self.norm(x)
        #x=self.mha(x0,x0,x)
        out_shape=(B,self.num_classes,D,H,W)
        embed_shape=(B,self.num_classes)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        x=self.softmax(x)
        return x

class MiniTransformer4(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3//4
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=2048,outdim=2048,num_heads=4)
        self.block2=DownBlock(dim=2048,hiddendim=512,outdim=256,num_heads=4)
        
        #self.block2=MyBlock(dim=4096,num_heads=4)
        #DownBlock(dim=2048,hiddendim=32,outdim=32,num_heads=8)
        self.block3=MyBlock(dim=256,num_heads=8)
        self.block4=MyBlock(dim=256,num_heads=8)
        self.block5=UpBlock(dim=256,hiddendim=2048,outdim=2048,num_heads=8)
        self.block6=UpBlock(dim=2048,hiddendim=16384,outdim=32768*2,num_heads=4)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(32768*2)
        self.pretrain=pretrain
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x0 = self.patch_embed(x)+self.pos_embed
        x1=self.block1(x0)
        x2=self.block2(x1)
        x3=self.block3(x2)
        x=self.block4(x3)+x2
        x=self.block5(x)+x1
        x=self.block6(x)
        x = self.norm(x)
        #x=self.mha(x0,x0,x)
        out_shape=(B,self.num_classes,D,H,W)
        embed_shape=(B,self.num_classes)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        x=self.softmax(x)
        return x
    
    def train(self,*args,**kwargs):
        if self.pretrain:
            self.softmax=nn.Softmax(dim=1)
            self.pretrain=False
        super(MiniTransformer4,self).train(*args,**kwargs)

class MiniTransformer5(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        self.pretrain=pretrain
        embed_dim=patch_size**3//2
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=DownBlock(dim=embed_dim,hiddendim=2048,outdim=2048,num_heads=8)
        
        self.block2=MyBlock(dim=2048,num_heads=8)
        #DownBlock(dim=2048,hiddendim=32,outdim=32,num_heads=8)
        #self.block3=MyBlock(dim=512,num_heads=8)
        #self.block4=MyBlock(dim=512,num_heads=8)
        #self.block5=UpBlock(dim=128,hiddendim=1024,outdim=2048,num_heads=8)
        self.block6=UpBlock(dim=2048,hiddendim=16384,outdim=32768*2,num_heads=4)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(32768*2)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x0 = self.patch_embed(x)
        x1=self.block1(x0)
        x2=self.block2(x1)+x1
        #x3=self.block3(x2)+x2
        #x=self.block4(x3)+x1
        #x=self.block5(x)+x0
        x=self.block6(x2)
        x = self.norm(x)
        #x=self.mha(x0,x0,x)
        out_shape=(B,self.num_classes,D,H,W)
        embed_shape=(B,self.num_classes)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        x=self.softmax(x)
        return x
    
    def train(self,*args,**kwargs):
        if self.pretrain:
            self.softmax=nn.Softmax(dim=1)
            self.pretrain=False
        super(MiniTransformer5,self).train(*args,**kwargs)

class MiniTransformer6(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=16, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        

        self.block1=UpBlock(dim=embed_dim,hiddendim=embed_dim*2,outdim=embed_dim*2,num_heads=2)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(embed_dim*2)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)
        self.pretrain=pretrain

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x0 = self.patch_embed(x)+self.pos_embed
        x1=self.block1(x0)
        x = self.norm(x1)
        #x=self.mha(x0,x0,x)
        out_shape=(B,self.num_classes,D,H,W)
        embed_shape=(B,self.num_classes)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        x=self.softmax(x)
        return x

    def train(self,*args,**kwargs):
        if self.pretrain:
            self.softmax=nn.Softmax(dim=1)
            self.pretrain=False
        super(MiniTransformer6,self).train(*args,**kwargs)

#これ    
class MiniTransformer7(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=16, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches
        patch_count=320*512*512//embed_dim
        self.num_classes=num_classes
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.block1=DownBlock2(dim=embed_dim,hiddendim=embed_dim//2,outdim=embed_dim//2,num_heads=4,patch_count=patch_count)
        #self.block2=MyBlock(dim=embed_dim//2,num_heads=4)
        self.block3=UpBlock2(dim=embed_dim//2,hiddendim=embed_dim,outdim=embed_dim,num_heads=2,patch_count=patch_count)
        self.norm=partial(nn.LayerNorm, eps=1e-6)([patch_count,embed_dim])
        #self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)
        self.sigmoid=nn.Sigmoid()
        self.pretrain=pretrain

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,C,D,H,W=x.shape
        x0 = self.norm(self.patch_embed(x)+self.pos_embed)
        x1=self.block1(x0)
        #x2=self.block2(x1)+x1
        x=self.block3(x1)

        #x=self.mha(x0,x0,x)
        out_shape=(B,1,D,H,W)
        embed_shape=(B,)+self.patch_embed.grid_size+self.patch_embed.patch_size
        x=torch.reshape(x,embed_shape)
        x=torch.reshape(x.permute(0,1,4,2,5,3,6),out_shape)
        #x=self.invconv(x)
        #x=self.softmax(x)
        x=self.sigmoid(x)
        if not self.pretrain:
            x=torch.cat([x,1-x],dim=1)
        return x

    def pre_train(self,*args,**kwargs):
        super().train(*args,**kwargs)
        self.pretrain=True

    def train(self,*args,**kwargs):
        if args:
            if args[0]:
                self.pretrain=False
        elif self.pretrain:
            self.pretrain=False
        super().train(*args,**kwargs)

class MiniTransformer8(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=16, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.block1=DownBlock(dim=embed_dim,hiddendim=64,outdim=64,num_heads=4)
        #self.block2=MyBlock(dim=64,num_heads=8)
        self.block3=UpBlock2(dim=64,hiddendim=embed_dim,outdim=embed_dim,num_heads=2)
        #self.block3=UpBlock(dim=embed_dim,hiddendim=embed_dim*2,outdim=embed_dim*2,num_heads=2)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(embed_dim)
        self.sigmoid=nn.Sigmoid()
        #self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)
        self.pretrain=pretrain

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,_,D,H,W=x.shape
        x= self.patch_embed(x)
        x=self.block1(x)
        #x2=self.block2(x1)+x1
        x=self.block3(x)
        x = self.norm(x)
        #x=self.mha(x0,x0,x)
        out_shape=(B,1,D,H,W)
        embed_shape=(B,1)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        #x=self.softmax(x)
        x=self.sigmoid(x)
        if not self.pretrain:
            x=torch.cat([x,1-x],dim=1)
        return x
    def pre_train(self,*args,**kwargs):
        super().train(*args,**kwargs)
        self.pretrain=True

    def train(self,*args,**kwargs):
        if self.pretrain:
            self.softmax=nn.Softmax(dim=1)
            self.pretrain=False
        super(MiniTransformer8,self).train(*args,**kwargs)
 
class Transformer1(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=16, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.block1=DownAttention(dim=embed_dim,out_dim=32,num_heads=2)
        self.block2=MyBlock(dim=32,num_heads=4)
        self.block3=nn.Linear(32,embed_dim,bias=False)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(embed_dim)
        self.sigmoid=nn.Sigmoid()
        #self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)
        self.pretrain=pretrain

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,_,D,H,W=x.shape
        x= self.patch_embed(x)
        x1=self.block1(x)
        x=self.block2(x1)+x1
        x=self.block3(x)
        x = self.norm(x)
        #x=self.mha(x0,x0,x)
        out_shape=(B,1,D,H,W)
        embed_shape=(B,1)+self.patch_embed.patch_size+self.patch_embed.grid_size
        x=torch.reshape(x.transpose(1,2),embed_shape)
        x=x.permute(0,1,2,5,3,6,4,7)
        x=x.reshape(out_shape)
        #x=self.invconv(x)
        #x=self.softmax(x)
        x=self.sigmoid(x)
        x=torch.cat([x,1-x],dim=1)
        return x

    def train(self,*args,**kwargs):
        if self.pretrain:
            self.softmax=nn.Softmax(dim=1)
            self.pretrain=False
        super(Transformer1,self).train(*args,**kwargs)
 
class Transformer2(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=16, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.block1=MyBlock(dim=embed_dim,num_heads=2,mlp_ratio=1,bias1=True,bias2=True,scale=500**-2,patch_count=20480)
        self.norm=partial(nn.LayerNorm, eps=1e-6)([20480,embed_dim])
        self.sigmoid=nn.Sigmoid()
        #self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)
        self.pretrain=pretrain

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,_,D,H,W=x.shape
        x= self.patch_embed(x)
        x = self.norm(x)
        x=self.block1(x)
        out_shape=(B,1,D,H,W)
        embed_shape=(B,)+self.patch_embed.grid_size+self.patch_embed.patch_size
        x=torch.reshape(x,embed_shape)
        x=torch.reshape(x.permute(0,1,4,2,5,3,6),out_shape)
        x=self.sigmoid(x)
        if not self.pretrain:
            x=torch.cat([x,1-x],dim=1)
        return x

    def pre_train(self,*args,**kwargs):
        super().train(*args,**kwargs)
        self.pretrain=True

    def train(self,*args,**kwargs):
        if args:
            if args[0]:
                self.pretrain=False
        elif self.pretrain:
            self.pretrain=False
        super(Transformer2,self).train(*args,**kwargs)

class Transformer3(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=16, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches
        patch_count=320*512*512//embed_dim
        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.block1=DownAttention2(dim=embed_dim,out_dim=32,num_heads=2)
        self.block2=MyBlock(dim=32,num_heads=2,mlp_ratio=1)
        self.block3=MyBlock(dim=32,num_heads=2,mlp_ratio=1)
        self.block4=nn.Linear(32,embed_dim,bias=False)
        self.norm1=partial(nn.LayerNorm, eps=1e-6)([patch_count,32])
        self.norm2=partial(nn.LayerNorm, eps=1e-6)([patch_count,embed_dim])
        self.sigmoid=nn.Sigmoid()
        #self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)
        self.pretrain=pretrain

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,_,D,H,W=x.shape
        x= self.norm2(self.patch_embed(x))
        x1=self.norm1(self.block1(x))    
        x2=self.block2(x1)+x1
        x=self.block3(x2)+x2
        x=self.block4(x)
        #x=self.mha(x0,x0,x)
        out_shape=(B,1,D,H,W)
        embed_shape=(B,)+self.patch_embed.grid_size+self.patch_embed.patch_size
        x=torch.reshape(x,embed_shape)
        x=torch.reshape(x.permute(0,1,4,2,5,3,6),out_shape)
        #x=self.invconv(x)
        #x=self.softmax(x)
        x=self.sigmoid(x)
        if not self.pretrain:
            x=torch.cat([x,1-x],dim=1)
        return x
    
    def pre_train(self,*args,**kwargs):
        super().train(*args,**kwargs)
        self.pretrain=True


    def train(self,*args,**kwargs):
        if self.pretrain:
            self.softmax=nn.Softmax(dim=1)
            self.pretrain=False
        super(Transformer3,self).train(*args,**kwargs)

class Transformer4(nn.Module):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, num_classes=2, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=False, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        #super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        super().__init__()
        #self.num_classes = num_classes
        embed_dim=patch_size**3
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches=self.patch_embed.num_patches

        self.num_classes=num_classes
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        patch_count=320*512*512//embed_dim

        self.block1=DownBlock2(dim=embed_dim,hiddendim=4096,outdim=1024,num_heads=4,patch_count=patch_count)
        self.block2=DownBlock2(dim=1024,hiddendim=256,outdim=256,num_heads=4,patch_count=patch_count)
        self.block3=MyBlock(dim=256,num_heads=8,mlp_ratio=1,patch_count=patch_count)
        self.block4=UpBlock2(dim=256,hiddendim=1024,outdim=1024,num_heads=4,patch_count=patch_count)
        self.block5=UpBlock2(dim=1024,hiddendim=4096,outdim=embed_dim,num_heads=4,patch_count=patch_count)
        self.norm=partial(nn.LayerNorm, eps=1e-6)([patch_count,embed_dim])
        self.sigmoid=nn.Sigmoid()
        #self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)
        self.pretrain=pretrain

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        B,_,D,H,W=x.shape
        x= self.patch_embed(x)
        x = self.norm(x)
        x1=self.block1(x)
        x2=self.block2(x1)
        x=self.block3(x2)+x2
        x=self.block4(x)+x1
        x=self.block5(x)
        out_shape=(B,1,D,H,W)
        embed_shape=(B,)+self.patch_embed.grid_size+self.patch_embed.patch_size
        x=torch.reshape(x,embed_shape)
        x=torch.reshape(x.permute(0,1,4,2,5,3,6),out_shape)
        #x=self.invconv(x)
        #x=self.softmax(x)
        x=self.sigmoid(x)
        if not self.pretrain:
            x=torch.cat([x,1-x],dim=1)
        return x
    
    def pre_train(self,*args,**kwargs):
        super().train(*args,**kwargs)
        self.pretrain=True


    def train(self,*args,**kwargs):
        if args:
            if args[0]:
                self.pretrain=False
        elif self.pretrain:
            self.pretrain=False
        super().train(*args,**kwargs)
  
 

if __name__=="__main__":
    x=torch.rand(1,1,320,512,512,requires_grad=True)
    #print(x.size())
    #model=VolumeTransformer5()
    #params = 0
    #for p in model.parameters():
    #    if p.requires_grad:
    #        params += p.numel()
    #print(params)

    
    device=torch.device("cuda:1")
    model=Transformer4()
    #.to(device)
    y=model(x)
    params = 0
    for p in model.parameters():
        if p.requires_grad:
            params += p.numel()
    print(params)
#
    #model=MiniTransformer7()
    #params = 0
    #for p in model.parameters():
    #    if p.requires_grad:
    #        params += p.numel()
    #print(params)

    #y=model.forward(x.to(device))
    #print(y.size())
    loss=BCE_with_DiceLoss(y,torch.zeros_like(y))
    loss.backward()
    print(x.grad)
#
    #print(loss)
    
    #print(y.shape)
    #print(y.max(),y.min())

