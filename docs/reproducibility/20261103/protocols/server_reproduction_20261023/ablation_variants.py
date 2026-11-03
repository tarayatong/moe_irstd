"""Explicit NUAA ablation definitions. No pretrained weights are used."""
import torch
from torch import nn
from torch.nn import functional as F
from model import mobile_mamba as mm
from model import conparitive_freq_module as fm
from model.model_mask_s_shape import DNANet
from model.model_DNANet import Res_CBAM_block
from model.load_param_data import load_param
from model.mask import SpatialSparseMoE

class WaveletAttentionFixed(fm.WaveletAttention):
    """Existing wavelet adaptation with image-local attention and valid IWT."""
    def __init__(self, channels):
        super().__init__(channels,32,4,8)
        self.attention.batch_first=True
        _,recon=mm.create_wavelet_filter('db1',channels//4,channels//4)
        self.iwt_filter=nn.Parameter(recon,requires_grad=False)
    def forward(self,x):
        x=self.pre_conv(x);b,c,h,w=x.shape;p=self.patch_size
        bands=self.wt_function(x)
        q=self.q_proj(self.layernorm1(self.patch_embed_x(x).flatten(2).transpose(1,2)))
        v=self.layernorm2(self.patch_embed_h(self.h_conv(bands)).flatten(2).transpose(1,2))
        z,_=self.attention(q,self.k_proj(v),self.v_proj(v),need_weights=False)
        z=z.reshape(b,h//p,w//p,c,p,p).permute(0,3,1,4,2,5).reshape(b,c,h,w)
        recon=F.conv_transpose2d(bands,self.iwt_filter,stride=2,groups=c)
        return self.out_conv(torch.cat([z,recon],dim=1))

class PatchFFTFixed(fm.PatchFFT):
    """Existing FFT-patch adaptation; attention operates within each image."""
    def __init__(self,channels,level):
        super().__init__(channels,32,4,layer=level)
        self.mha.batch_first=True
    def forward(self,x):
        x=self.pre_process(x);b,c,h,w=x.shape;p=self.patch_size
        tiles=fm.extract_patches(x,p).reshape(b,c,-1,p*p).transpose(1,2)
        z=torch.fft.fft2(tiles.reshape(-1,c,p,p)).real
        z=self.conv2(F.relu(self.conv1(self.patch_embed(z))))
        z=self.layernorm(z.reshape(b,-1,self.embed_dim))
        z,_=self.mha(self.query(z),self.key(z),self.value(z),need_weights=False)
        z=z.reshape(b,h//p,w//p,c,p,p).permute(0,3,1,4,2,5).reshape(b,c,h,w)
        return self.output_conv(z)

class FixedMixture(nn.Module):
    def __init__(self,experts):
        super().__init__();self.experts=nn.ModuleList(experts);self.last_aux_loss=None
    def forward(self,x):return sum(e(x) for e in self.experts)/len(self.experts)

class ChannelMixture(nn.Module):
    """Per-image/per-channel routing, normalized across the expert dimension."""
    def __init__(self,channels,experts,k,noise_scale):
        super().__init__();self.experts=nn.ModuleList(experts);self.e=len(experts);self.k=min(k,self.e);self.noise_scale=noise_scale
        self.routing_layer=nn.Conv2d(channels,channels*self.e,1,groups=channels)
        self.noise_layer=nn.Conv2d(channels,channels*self.e,1,groups=channels)
        self.last_aux_loss=None
    def forward(self,x):
        b,c,h,w=x.shape;pooled=x.mean((2,3),keepdim=True)
        logits=self.routing_layer(pooled).reshape(b,c,self.e)
        if self.training and self.noise_scale>0:
            logits=logits+self.noise_scale*torch.randn_like(logits)*F.softplus(self.noise_layer(pooled).reshape(b,c,self.e))
        val,idx=logits.topk(self.k,dim=-1)
        sparse=torch.full_like(logits,float('-inf')).scatter(-1,idx,val);weight=sparse.softmax(-1)
        out=torch.zeros_like(x)
        for i,expert in enumerate(self.experts):
            mask=(idx==i).any(-1).unsqueeze(-1).unsqueeze(-1)
            if mask.any():out=out+expert(x*mask)*weight[:,:,i,None,None]
        if self.training:
            f=F.one_hot(idx,num_classes=self.e).float().sum(2).mean((0,1));p=weight.mean((0,1))
            self.last_aux_loss=self.e*(f*p).sum()/self.k
        else:self.last_aux_loss=None
        return out

def build_model(cfg):
    filters,blocks=load_param('three','resnet_18')
    net=DNANet(1,3,Res_CBAM_block,blocks,filters,moe_stages=[bool(v) for v in cfg['moe_stages']],dilations=cfg['dilations'],noise_scale=cfg['noise_scale'],patch_size=cfg['patch_size'],top_k=cfg['top_k'],input_routing=cfg['input_routing'])
    notes=[]
    for name,m in list(net.named_modules()):
        if not isinstance(m,mm.MobileMambaModule):continue
        experts=list(m.spacial_moe.experts);dim=m.dim;level=int(name.split('.')[2])
        variant=cfg.get('expert_variant')
        if variant:
            if variant=='single_spatial':m.local_op=mm.DWConv2d_BN_ReLU(dim,dim,3)
            frequencies=[]
            if variant=='multi_low':frequencies=[experts[1]]
            elif variant=='multi_high':frequencies=[experts[0]]
            elif variant not in ('single_spatial','multi_spatial'):raise ValueError(variant)
            m.global_op=frequencies;experts=frequencies+[m.local_op,nn.Identity()]
        freq=cfg.get('frequency_variant')
        if freq:
            if freq=='HLSR':op=fm.FourierCombineNet(dim)
            elif freq=='FreqODEs':op=WaveletAttentionFixed(dim)
            elif freq=='FAM':op=PatchFFTFixed(dim,level)
            elif freq=='FDD':op=fm.FrequencySeparation(dim,cfg['base_size']//2**level,cfg['base_size']//2**level)
            elif freq=='LGFW':op=fm.Freq_block(dim)
            else:raise ValueError(freq)
            m.global_op=[op];experts=[op,m.local_op,nn.Identity()]
        router=cfg.get('router_variant')
        if router=='fixed':new=FixedMixture(experts)
        elif router in ('channel_weighted','channel_gated'):
            new=ChannelMixture(dim,experts,len(experts) if router.endswith('weighted') else cfg['top_k'],cfg['noise_scale'])
        elif router=='pixel_weighted':
            new=SpatialSparseMoE(dim,len(experts),len(experts),dim,nn.ModuleList(experts),noise_scale=cfg['noise_scale'],patch_size=1,input_routing=cfg['input_routing'])
        elif variant or freq:
            new=SpatialSparseMoE(dim,len(experts),min(cfg['top_k'],len(experts)),dim,nn.ModuleList(experts),noise_scale=cfg['noise_scale'],patch_size=1,input_routing=cfg['input_routing'])
        else:continue
        m.spacial_moe=new
        notes.append(dict(module=name,level=level,experts=[type(e).__name__ for e in experts],router=type(new).__name__))
    return net,notes

def auxiliary_loss(net):
    items=[m.last_aux_loss for m in net.modules() if isinstance(m,(SpatialSparseMoE,ChannelMixture)) and m.last_aux_loss is not None]
    return torch.stack(items).mean() if items else None
