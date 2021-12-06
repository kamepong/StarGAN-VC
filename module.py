# Copyright 2021 Hirokazu Kameoka

import torch
import torch.nn as nn

def calc_padding(kernel_size, dilation, causal, stride=1):
    if causal:
        padding = (kernel_size-1)*dilation+1-stride
    else:
        padding = ((kernel_size-1)*dilation+1-stride)//2
    return padding

class LinearWN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
        super(LinearWN, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        #nn.init.xavier_normal_(self.linear_layer.weight,gain=0.1)
        self.linear_layer = nn.utils.weight_norm(self.linear_layer)

    def forward(self, x):
        return self.linear_layer(x)

class ConvGLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, sd, normtype='IN'):
        super(ConvGLU1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch*2, ks, stride=sd, padding=(ks-sd)//2)
        #nn.init.xavier_normal_(self.conv1.weight,gain=0.1)
        if normtype=='BN':
            self.norm1 = nn.BatchNorm1d(out_ch*2)
        elif normtype=='IN':
            self.norm1 = nn.InstanceNorm1d(out_ch*2)
        elif normtype=='LN':
            self.norm1 = nn.LayerNorm(out_ch*2)

        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.normtype = normtype

    def __call__(self, x):
        h = self.conv1(x)
        if self.normtype=='BN' or self.normtype=='IN':
            h = self.norm1(h)
        elif self.normtype=='LN':
            B, D, N = h.shape
            h = h.permute(0,2,1).reshape(-1,D)
            h = self.norm1(h)
            h = h.reshape(B,N,D).permute(0,2,1)
        h_l, h_g = torch.split(h, h.shape[1]//2, dim=1)
        h = h_l * torch.sigmoid(h_g)
        
        return h

class DeconvGLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, sd, normtype='IN'):
        super(DeconvGLU1D, self).__init__()
        self.conv1 = nn.ConvTranspose1d(
            in_ch, out_ch*2, ks, stride=sd, padding=(ks-sd)//2)
        #nn.init.xavier_normal_(self.conv1.weight,gain=0.1)
        if normtype=='BN':
            self.norm1 = nn.BatchNorm1d(out_ch*2)            
        elif normtype=='IN':
            self.norm1 = nn.InstanceNorm1d(out_ch*2)
        elif normtype=='LN':
            self.norm1 = nn.LayerNorm(out_ch*2)

        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.normtype = normtype

    def __call__(self, x):
        h = self.conv1(x)
        if self.normtype=='BN' or self.normtype=='IN':
            h = self.norm1(h)
        elif self.normtype=='LN':
            B, D, N = h.shape
            h = h.permute(0,2,1).reshape(-1,D)
            h = self.norm1(h)
            h = h.reshape(B,N,D).permute(0,2,1)
        h_l, h_g = torch.split(h, h.shape[1]//2, dim=1)
        h = h_l * torch.sigmoid(h_g)
        
        return h

class PixelShuffleGLU1D(nn.Module):
    def __init__(self, in_ch, out_ch, ks, sd, normtype='IN'):
        super(PixelShuffleGLU1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_ch, out_ch*2*sd, ks, stride=1, padding=(ks-1)//2)
        self.r = sd
        if normtype=='BN':
            self.norm1 = nn.BatchNorm1d(out_ch*2)
        elif normtype=='IN':
            self.norm1 = nn.InstanceNorm1d(out_ch*2)
        self.conv1 = nn.utils.weight_norm(self.conv1)
        self.normtype = normtype

    def __call__(self, x):
        h = self.conv1(x)
        N, pre_ch, pre_len = h.shape
        r = self.r
        post_ch = pre_ch//r
        post_len = pre_len * r
        h = torch.reshape(h, (N, r, post_ch, pre_len))
        h = h.permute(0,2,3,1)

        h = torch.reshape(h, (N, post_ch, post_len))
        if self.normtype=='BN' or self.normtype=='IN':
            h = self.norm1(h)
        h_l, h_g = torch.split(h, h.shape[1]//2, dim=1)
        h = h_l * torch.sigmoid(h_g)
        
        return h

def concat_dim1(x,y):
    assert x.shape[0] == y.shape[0]
    if torch.Tensor.dim(x) == 3:
        y0 = torch.unsqueeze(y,2)
        N, n_ch, n_t = x.shape
        yy = y0.repeat(1,1,n_t)
        h = torch.cat((x,yy), dim=1)
    elif torch.Tensor.dim(x) == 4:
        y0 = torch.unsqueeze(torch.unsqueeze(y,2),3)
        N, n_ch, n_q, n_t = x.shape
        yy = y0.repeat(1,1,n_q,n_t)
        h = torch.cat((x,yy), dim=1)
    return h

def concat_dim2(x,y):
    assert x.shape[0] == y.shape[0]
    if torch.Tensor.dim(x) == 3:
        y0 = torch.unsqueeze(y,1)
        N, n_t, n_ch = x.shape
        yy = y0.repeat(1,n_t,1)
        h = torch.cat((x,yy), dim=2)
    elif torch.Tensor.dim(x) == 2:
        h = torch.cat((x,y), dim=1)
    return h