import torch
from torch import nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer,Block,Attention
from timm.models.layers import DropPath,Mlp
import os
from config import CONFIG
from functools import partial


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

def load_model(path,device,model_type):
    model=eval(f"{model_type}()")
    model.load_state_dict(torch.load(path,map_location=device))
    return model.to(device)

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
        #print(self.grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.flatten = flatten
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        #self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNDim
        #print(x.shape)
        #x = self.norm(x)
        #print(x.shape)
        return x

class VolumePatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    def __init__(self, img_size=IMAGE_SIZE,img_depth=MAX_DEPTH, patch_size=BLOCK_SIZE, in_chans=2, embed_dim=BLOCK_SIZE**3, norm_layer=None, flatten=True):
        super().__init__()
        img_size = (img_depth,img_size,img_size)
        patch_size = (patch_size,patch_size,patch_size)
        #print(img_size,patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1],img_size[2]//patch_size[2])
        #print(self.grid_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        #print(self.num_patches)
        self.flatten = flatten
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        #self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCDHW -> BNDim
        return x

class MyAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        #print(x.shape)
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

class MyBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MyAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        #print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        x = self.drop_path(self.norm1(self.attn(x)+x))
        #print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = x+self.drop_path(self.norm2(self.mlp(x)))
        return x

class VolumeBlock(nn.Module):
    def __init__(self, dim, hiddendim,outdim,num_heads, qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MyAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(outdim)
        #mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=hiddendim,out_features=outdim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        #print(x.shape)
        #x = x + self.drop_path(self.norm1(self.attn(x)))
        #print(x.shape)
        x = self.drop_path(self.norm1(self.attn(x)+x))
        #print(x.shape)
        #x = x + self.drop_path(self.norm2(self.mlp(x)))
        x = self.drop_path(self.norm2(self.mlp(x)))
        return x

class ThreeDimensionalTransformer(VisionTransformer):
    def __init__(self, img_size=INPUT_IMAGE_SIZE, patch_size=BLOCK_SIZE, in_chans=1, out_chans=2, embed_dim=EMBED_DIM, depth=DEPTH,
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        self.out_chans=out_chans
        self.depth=depth
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            MyBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.representation=nn.Linear(self.patch_embed.num_patches,1)
        #self.act_layer=act_layer()
        #self.output=nn.Linear(embed_dim,patch_size**3*out_chans)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        """saved=[]
        for i,block in enumerate(self.blocks):
            if i<self.depth//2:
                saved.append(x)
                x=block(x)
            else:
                x=block(x)+saved[self.depth-i-1]"""
        x = self.norm(x)
        #print(x.shape)->B N ED
        x=self.representation(x.transpose(1,2))
        #x=self.act_layer(x)
        #print(x.shape)->B ED 1
        #x=self.output(x.transpose(1,2))
        #x=super().forward(x)
        #print(x.shape) B 1 W^3xC
        B,_,_=x.shape
        output_shape=(B,self.out_chans)+self.patch_embed.patch_size
        x=torch.reshape(x,output_shape)
        x=self.softmax(x)
        return x

class ThreeDimensionalUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, width_multiplier=1, trilinear=True, use_ds_conv=False,pretrain=False):
        super(ThreeDimensionalUNet, self).__init__()
        _channels = (32, 64, 128, 256, 512)
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
        #self.down4 = Down(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        #self.up1 = Up(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2 = Up(self.channels[3], self.channels[2] // factor, trilinear)
        self.up3 = Up(self.channels[2], self.channels[1] // factor, trilinear)
        self.up4 = Up(self.channels[1], self.channels[0], trilinear)
        self.outc = OutConv(self.channels[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        #print(x1.shape)
        x2 = self.down1(x1)
        #print(x2.shape)
        x3 = self.down2(x2)
        #print(x3.shape)
        x4 = self.down3(x3)
        #print(x4.shape)
        #x5 = self.down4(x4)
        #print(x5.shape)
        #x = self.up1(x5, x4)
        #print(x.shape)
        x = self.up2(x4, x3)
        #print(x.shape)
        x = self.up3(x, x2)
        #print(x.shape)
        #x = self.up4(x, x1)
        #print(x.shape)
        logits = self.outc(x)
        return logits

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

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2,pad=0)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,pad=0)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        #print(x1.shape)
        #print([diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2,diffZ // 2, diffZ - diffZ // 2,])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2,])
        #print(x1.shape,x2.shape)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
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

class VolumeTransformer(VisionTransformer):
    def __init__(self, img_size=IMAGE_SIZE, patch_size=32, in_chans=1, out_chans=2, embed_dim=EMBED_DIM, 
                 num_heads=NUM_HEADS, mlp_ratio=0.25, qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=VolumePatchEmbed,norm_layer=nn.LayerNorm,act_layer=nn.ReLU,pretrain=False):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,num_heads=num_heads,embed_layer=embed_layer)
        self.out_chans=out_chans
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))

        self.block1=VolumeBlock(dim=1024,hiddendim=512,outdim=512,num_heads=num_heads)
        self.block2=VolumeBlock(dim=512,hiddendim=256,outdim=256,num_heads=num_heads)
        self.block3=VolumeBlock(dim=256,hiddendim=128,outdim=128,num_heads=num_heads)
        self.block4=VolumeBlock(dim=128,hiddendim=128,outdim=256,num_heads=num_heads)
        self.block5=VolumeBlock(dim=256,hiddendim=256,outdim=512,num_heads=num_heads)
        self.block6=VolumeBlock(dim=512,hiddendim=1024,outdim=4096,num_heads=num_heads)
        self.block7=VolumeBlock(dim=4096,hiddendim=8192,outdim=65536,num_heads=num_heads)
        self.norm=partial(nn.LayerNorm, eps=1e-6)(65536)
        #self.representation=nn.Linear(self.patch_embed.num_patches,1)
        #self.act_layer=act_layer()
        #self.output=nn.Linear(embed_dim,patch_size**3*out_chans)
        self.softmax=nn.Softmax(dim=1) if not pretrain else nn.Conv3d(2,1,1)

    def forward(self, x):
        #x = self.forward_features(x)
        #print(x.shape)
        x = self.patch_embed(x)
        #print(x.shape)
        x = self.pos_drop(x + self.pos_embed)
        #x = self.blocks(x)
        x1=self.block1(x)
        x2=self.block2(x1)
        x=self.block3(x2)
        x=self.block4(x)+x2
        x=self.block5(x)+x1
        x=self.block6(x)
        x=self.block7(x)
        x = self.norm(x)
        #x=self.representation(x.transpose(1,2))
        B,_,_=x.shape
        output_shape=(B,self.out_chans)+self.patch_embed.img_size
        #print(output_shape)
        x=torch.reshape(x,output_shape)
        x=self.softmax(x)
        return x


if __name__=="__main__":
    x=torch.rand(1,1,MAX_DEPTH,IMAGE_SIZE,IMAGE_SIZE)
    model=VolumeTransformer()
    y=model.forward(x)
    print(y.shape)
    print(y.max(),y.min())

