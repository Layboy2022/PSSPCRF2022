

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import math
from torch.optim.lr_scheduler import StepLR




seq_len = 700
featureset = range(0,63)
targetset = range(63,72)
INPUTS_SIZE = len(featureset)
LABELS_NUM = 3

learning_rate = 0.003
USE_GPU = True
dropout_rateLRI = 0.2
dropout_rate_Conv = 0.2
dropout_rateRNN = 0.2
LAM = 0.001
kernel_size = [3, 1]
out_channels = 64

LRI_SIZE = 512
LRI_SIZE1 = 256
HIDDEN_SIZE = 400
N_LAYER = 2
bidirectional = True


N_EPOCHS = 400
pad_len = int((kernel_size[0]-1)/2)

BATCH_SIZE = 64
BATCH_SIZE_TEST = 128



class NameDataset(Dataset):
    def __init__(self, is_train_set=0):
        traindata = np.load(r'.\data\traindata.npy')
        validata = np.load(r'.\data\validata.npy')
        testdata = np.load(r'.\data\testdata.npy')
        CB513data = np.load(r'.\data\CB513data.npy')
        CASP10data = np.load(r'.\data\CASP10data.npy')
        CASP11data = np.load(r'.\data\CASP11data.npy')

        if is_train_set == 0:
            data = [x for i, x in enumerate(traindata)]
        elif is_train_set == 1:
            data = [x for i, x in enumerate(testdata)]
        elif is_train_set == 2:
            data = [x for i, x in enumerate(validata)]
        elif is_train_set == 3:
            data = [x for i, x in enumerate(CB513data)]
        elif is_train_set == 4:
            data = [x for i, x in enumerate(CASP10data)]
        else:
            data = [x for i, x in enumerate(CASP11data)]

        self.data = data
        self.len = len(self.data)

    def __getitem__(self, index):
        return torch.from_numpy(self.data[index]).float()

    def __len__(self):
        return self.len


trainset = NameDataset(is_train_set=0)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=1)
testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=False)
valiset = NameDataset(is_train_set=2)
valiloader = DataLoader(valiset, batch_size=BATCH_SIZE_TEST, shuffle=False)
testset513 = NameDataset(is_train_set=3)
testloader513 = DataLoader(testset513, batch_size=BATCH_SIZE_TEST, shuffle=False)
testsetCASP10 = NameDataset(is_train_set=4)
testloaderCASP10 = DataLoader(testsetCASP10, batch_size=BATCH_SIZE_TEST, shuffle=False)
testsetCASP11 = NameDataset(is_train_set=5)
testloaderCASP11 = DataLoader(testsetCASP11, batch_size=BATCH_SIZE_TEST, shuffle=False)

# dropout
dropoutLRI = nn.Dropout(p=dropout_rateLRI)
dropoutRNN = nn.Dropout(p=dropout_rateRNN)
dropoutConv = nn.Dropout(p=dropout_rate_Conv)


class CNNLayer(nn.Module):
    def __init__(self):
        super(CNNLayer, self).__init__()
        self.conv = torch.nn.Sequential(
            # （batch_size, out_channels, seq_len, 1）
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=(1, 1),
                            padding=(pad_len, 0)),
            torch.nn.ReLU(),
        )
        self.layernorm = nn.LayerNorm(out_channels)
        self.ReLu = torch.nn.ReLU()

    def forward(self, x):
        # （batch_size, out_channels, seq_len, 1）
        x = self.layernorm(x.permute(0, 2, 3, 1)+self.conv(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.ReLu(x)
        if self.training:
            x = dropoutConv(x)
        return x


class RNNLayer(nn.Module):
    def __init__(self):
        super(RNNLayer, self).__init__()
        self.n_directions = 2 if bidirectional else 1
        # input: (seqLen,batchSize,input_size)   output: (seqLen,batchSize,hiddenSize*nDirections)
        self.gru = nn.GRU(out_channels, HIDDEN_SIZE, N_LAYER, bidirectional=bidirectional, dropout=dropout_rateRNN)

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(N_LAYER * self.n_directions, batch_size, HIDDEN_SIZE)
        return create_tensor(hidden)

    def forward(self, x):
        batch_size = x.size(0)
        output, hidden = self.gru(x.permute(1, 0, 2), self._init_hidden(batch_size))
        # shape: (seq_len, batch, :) -> (batch, seq_len, :)
        output = output.permute(1, 0, 2)
        if self.training:
            output = dropoutRNN(output)
        output = torch.cat((x, output), 2)

        return output

class CRNNmodel(nn.Module):
    def __init__(self):
        super(CRNNmodel, self).__init__()

        self.n_directions = 2 if bidirectional else 1

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, out_channels, kernel_size=[1, INPUTS_SIZE], stride=(1, 1)),
            torch.nn.ReLU(),
        )
        self.layers = nn.ModuleList([CNNLayer() for _ in range(1)])
        self.RNNlayer = RNNLayer()

        self.FC = torch.nn.Sequential(
             torch.nn.Linear(out_channels+INPUTS_SIZE + (HIDDEN_SIZE * self.n_directions), LRI_SIZE),
             torch.nn.ReLU(),
             torch.nn.Linear(LRI_SIZE, LRI_SIZE1),
             torch.nn.ReLU(),
        )

        self.FC0 = torch.nn.Linear(LRI_SIZE1, LABELS_NUM)

        self.layernorm = nn.LayerNorm(out_channels)
        self.FCRNN0 = torch.nn.Linear(INPUTS_SIZE, out_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = x
        #  (batch_size, 1, seq_len, INPUTS_SIZE)
        x = x.unsqueeze(1)
        #  (batch_size, out_channels, seq_len, 1)
        x = self.conv1(x)
        x = self.layernorm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        for layer in self.layers:
            x = layer(x)
        # （batch_size, seq_len, 1, out_channels）
        x = x.permute(0, 2, 3, 1).reshape(batch_size, seq_len, -1)
        #(batch_size, seq_len, out_channels+ hidden_size * nDirections)
        x = self.RNNlayer(x)
        x = torch.cat((x, x1), 2)
        x1 = self.FC(x)
        if self.training:
            x1 = dropoutLRI(x1)
        x = self.FC0(x1)
        return x, x1

def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor

def make_tensors1(data):
    inputs = data[:, 0:seq_len, featureset]
    target = data[:, 0:seq_len, targetset]
    target0 = target[:, 0:seq_len, :4]
    target0[:,:,0] = target[:,:,0]+target[:,:,6]+target[:,:,7] # L,S,T
    target0[:,:,1] = target[:,:,1]+target[:,:,2] # B,E
    target0[:,:,2] = target[:,:,3]+target[:,:,4]+target[:,:,5] # G,I,H
    target0[:,:,3] = target[:,:,8]
    target=target0
    targ = target.max(dim=2)[1]
    targnum = (~np.isin(targ, np.array([3]))).sum(axis=1)
    inputs = [inputs[idx, np.append(range(targnum[idx]-1, max(-1, targnum[idx]-1-seq_len), -1),
                                    range(targnum[idx], seq_len+1+max(-1, targnum[idx]-1-seq_len))), :].numpy()
               for idx in range(inputs.shape[0])]
    inputs = torch.tensor(np.array(inputs))
    target = [target[idx, np.append(range(targnum[idx]-1, max(-1, targnum[idx]-1-seq_len), -1),
                                    range(targnum[idx], seq_len+1+max(-1, targnum[idx]-1-seq_len))), :].numpy()
               for idx in range(inputs.shape[0])]
    target = torch.tensor(np.array(target))
    targ = target.max(dim=2)[1]
    targlc = targ.reshape(-1)
    targidx = ~np.isin(targlc, np.array([3]))
    target = target[:, :, 0:LABELS_NUM]

    return create_tensor(inputs), create_tensor(targlc), targidx, create_tensor(target)

def make_tensors(data):
    inputs = data[:, 0:seq_len, featureset]
    target = data[:, 0:seq_len, targetset]
    target0 = target[:, 0:seq_len, :4]
    target0[:,:,0] = target[:,:,0]+target[:,:,6]+target[:,:,7] # L,S,T
    target0[:,:,1] = target[:,:,1]+target[:,:,2] # B,E
    target0[:,:,2] = target[:,:,3]+target[:,:,4]+target[:,:,5] # G,I,H
    target0[:,:,3] = target[:,:,8]
    target=target0
    targ = target.max(dim=2)[1]
    targlc = targ.reshape(-1)
    targidx = ~np.isin(targlc, np.array([3]))
    target = target[:, :, 0:LABELS_NUM]

    return create_tensor(inputs), create_tensor(targlc), targidx, create_tensor(target)


CRNNmodel = CRNNmodel()
# training =False
CRNNmodel.eval()

if USE_GPU:
    device = torch.device("cuda:0")
    CRNNmodel.to(device)

outputME=torch.ones(0,seq_len,LABELS_NUM)
targetME=torch.ones(0,seq_len)
targidxME=np.ones((0,seq_len))
outputMT513=torch.ones(0,seq_len,LABELS_NUM)
targetMT513=torch.ones(0,seq_len)
targidxMT513=np.ones((0,seq_len))
outputMT10=torch.ones(0,seq_len,LABELS_NUM)
targetMT10=torch.ones(0,seq_len)
targidxMT10=np.ones((0,seq_len))
outputMT11=torch.ones(0,seq_len,LABELS_NUM)
targetMT11=torch.ones(0,seq_len)
targidxMT11=np.ones((0,seq_len))

bnixu = [False, True]

bdir = [r'.\parameter\modelpara_forward_Q3.pth', r'.\parameter\modelpara_reversed_Q3.pth']

for iiddxx in range(2):
    nixu=bnixu[iiddxx]
    dir=bdir[iiddxx]
    checkpoint = torch.load(dir)
    CRNNmodel.load_state_dict(checkpoint['net'])
    with torch.no_grad():
        for i, data in enumerate(testloader, 1):
            # inputs (batch_size, seq_len, INPUTS_SIZE)
            # target (batch_size, seq_len, LABELS_NUM)
            if nixu:
                inputs, _, targidx, target = make_tensors1(data)
            else:
                inputs, _, targidx, target = make_tensors(data)
            output, _ = CRNNmodel(inputs)
            output = output
            target = target.max(dim=2)[1]
            targidx = targidx.reshape(-1, seq_len)
            outputME = torch.cat((outputME,output.cpu()),0)
            targetME = torch.cat((targetME,target.cpu()),0)
            targidxME = np.append(targidxME,targidx,0)

        for i, data in enumerate(testloader513, 1):
            # inputs (batch_size, seq_len, INPUTS_SIZE)
            # target (batch_size, seq_len, LABELS_NUM)
            if nixu:
                inputs, _, targidx, target = make_tensors1(data)
            else:
                inputs, _, targidx, target = make_tensors(data)
            output, _ = CRNNmodel(inputs)
            output = output
            target = target.max(dim=2)[1]
            targidx = targidx.reshape(-1, seq_len)
            outputMT513 = torch.cat((outputMT513,output.cpu()),0)
            targetMT513 = torch.cat((targetMT513,target.cpu()),0)
            targidxMT513 = np.append(targidxMT513,targidx,0)

        for i, data in enumerate(testloaderCASP10, 1):
            # inputs (batch_size, seq_len, INPUTS_SIZE)
            # target (batch_size, seq_len, LABELS_NUM)
            if nixu:
                inputs, _, targidx, target = make_tensors1(data)
            else:
                inputs, _, targidx, target = make_tensors(data)
            output, _ = CRNNmodel(inputs.float())
            output = output
            target = target.max(dim=2)[1]
            targidx = targidx.reshape(-1, seq_len)
            outputMT10 = torch.cat((outputMT10,output.cpu()),0)
            targetMT10 = torch.cat((targetMT10,target.cpu()),0)
            targidxMT10 = np.append(targidxMT10,targidx,0)

        for i, data in enumerate(testloaderCASP11, 1):
            # inputs (batch_size, seq_len, INPUTS_SIZE)
            # target (batch_size, seq_len, LABELS_NUM)
            if nixu:
                inputs, _, targidx, target = make_tensors1(data)
            else:
                inputs, _, targidx, target = make_tensors(data)
            output, _ = CRNNmodel(inputs.float())
            output = output
            target = target.max(dim=2)[1]
            targidx = targidx.reshape(-1, seq_len)
            outputMT11 = torch.cat((outputMT11,output.cpu()),0)
            targetMT11 = torch.cat((targetMT11,target.cpu()),0)
            targidxMT11 = np.append(targidxMT11,targidx,0)
dir=''


ifnorm = False
if ifnorm:
    outputME=torch.softmax(outputME,dim=2)

perc = torch.zeros(3,4)

bseq_len = 700

INDEX=range(0,700)

lenn = int(outputME.shape[0]/2)
outputME0=torch.zeros(lenn,seq_len,LABELS_NUM)
targetME0=torch.zeros(lenn,seq_len)
targidxME0=np.zeros((lenn,seq_len))
correct = 0
total = 0
for idx in range(lenn):
    sumi = int(targidxME[idx, :].sum())
    sumi0 = min(bseq_len,sumi)
    sumi2 = int((sumi-sumi0)/2)
    sumi1 = sumi-sumi0-sumi2
    outputME0[idx,:,:] = torch.cat((outputME[idx,0:sumi1,:],\
                                  outputME[idx,sumi1:sumi-sumi2,:]+outputME[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1),:],\
                                  outputME[idx+lenn,range(sumi2-1,-1,-1),:], outputME[idx,int(sumi):,:]),0)
    targetME0[idx,:] = torch.cat((targetME[idx,0:sumi1], targetME[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1)],\
                                  targetME[idx+lenn,range(sumi2-1,-1,-1)], targetME[idx,int(sumi):]),0)
    targidxME0[idx,:sumi1+sumi0] = np.append(targidxME[idx,0:sumi1], targidxME[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1)],0)
    targidxME0[idx,sumi1+sumi0:] = np.append(targidxME[idx+lenn,range(sumi2-1,-1,-1)], targidxME[idx,int(sumi):],0)
outputME0=outputME0.max(dim=2)[1]
if (targetME0-targetME[0:lenn,:]).max()>0:
    #fault
    1/0
pred = outputME0[:,INDEX].reshape(-1)[targidxME0[:,INDEX].reshape(-1)>0]
targ = targetME0[:,INDEX].reshape(-1)[targidxME0[:,INDEX].reshape(-1)>0]
perc[0,0] = pred.eq(targ).sum().item()
perc[1,0] = len(targ)
perc[2,0] = pred.eq(targ).sum().item()/len(targ)



lenn = int(outputMT513.shape[0]/2)
outputMT5130=torch.zeros(lenn,seq_len,LABELS_NUM)
targetMT5130=torch.zeros(lenn,seq_len)
targidxMT5130=np.zeros((lenn,seq_len))
correct = 0
total = 0
for idx in range(lenn):
    sumi = int(targidxMT513[idx, :].sum())
    sumi0 = min(bseq_len,sumi)
    sumi2 = int((sumi-sumi0)/2)
    sumi1 = sumi-sumi0-sumi2
    outputMT5130[idx,:,:] = torch.cat((outputMT513[idx,0:sumi1,:],\
                                  outputMT513[idx,sumi1:sumi-sumi2,:]+outputMT513[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1),:],\
                                  outputMT513[idx+lenn,range(sumi2-1,-1,-1),:], outputMT513[idx,int(sumi):,:]),0)
    targetMT5130[idx,:] = torch.cat((targetMT513[idx,0:sumi1], targetMT513[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1)],\
                                  targetMT513[idx+lenn,range(sumi2-1,-1,-1)], targetMT513[idx,int(sumi):]),0)
    targidxMT5130[idx,:sumi1+sumi0] = np.append(targidxMT513[idx,0:sumi1], targidxMT513[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1)],0)
    targidxMT5130[idx,sumi1+sumi0:] = np.append(targidxMT513[idx+lenn,range(sumi2-1,-1,-1)], targidxMT513[idx,int(sumi):],0)
outputMT5130=outputMT5130.max(dim=2)[1]
if (targetMT5130-targetMT513[0:lenn,:]).max()>0:
    1/0
pred = outputMT5130[:,INDEX].reshape(-1)[targidxMT5130[:,INDEX].reshape(-1)>0]
targ = targetMT5130[:,INDEX].reshape(-1)[targidxMT5130[:,INDEX].reshape(-1)>0]
perc[0,1] = pred.eq(targ).sum().item()
perc[1,1] = len(targ)
perc[2,1] = pred.eq(targ).sum().item()/len(targ)


lenn = int(outputMT10.shape[0]/2)
outputMT100=torch.zeros(lenn,seq_len,LABELS_NUM)
targetMT100=torch.zeros(lenn,seq_len)
targidxMT100=np.zeros((lenn,seq_len))
correct = 0
total = 0
for idx in range(lenn):
    sumi = int(targidxMT10[idx, :].sum())
    sumi0 = min(bseq_len,sumi)
    sumi2 = int((sumi-sumi0)/2)
    sumi1 = sumi-sumi0-sumi2
    outputMT100[idx,:,:] = torch.cat((outputMT10[idx,0:sumi1,:],\
                                  outputMT10[idx,sumi1:sumi-sumi2,:]+outputMT10[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1),:],\
                                  outputMT10[idx+lenn,range(sumi2-1,-1,-1),:], outputMT10[idx,int(sumi):,:]),0)
    targetMT100[idx,:] = torch.cat((targetMT10[idx,0:sumi1], targetMT10[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1)],\
                                  targetMT10[idx+lenn,range(sumi2-1,-1,-1)], targetMT10[idx,int(sumi):]),0)
    targidxMT100[idx,:sumi1+sumi0] = np.append(targidxMT10[idx,0:sumi1], targidxMT10[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1)],0)
    targidxMT100[idx,sumi1+sumi0:] = np.append(targidxMT10[idx+lenn,range(sumi2-1,-1,-1)], targidxMT10[idx,int(sumi):],0)
outputMT100=outputMT100.max(dim=2)[1]
if (targetMT100-targetMT10[0:lenn,:]).max()>0:
    1/0
pred = outputMT100.reshape(-1)[targidxMT100.reshape(-1)>0]
targ = targetMT100.reshape(-1)[targidxMT100.reshape(-1)>0]
perc[0,2] = pred.eq(targ).sum().item()
perc[1,2] = len(targ)
perc[2,2] = pred.eq(targ).sum().item()/len(targ)


lenn = int(outputMT11.shape[0]/2)
outputMT110=torch.zeros(lenn,seq_len,LABELS_NUM)
targetMT110=torch.zeros(lenn,seq_len)
targidxMT110=np.zeros((lenn,seq_len))
correct = 0
total = 0
for idx in range(lenn):
    sumi = int(targidxMT11[idx, :].sum())
    sumi0 = min(bseq_len,sumi)
    sumi2 = int((sumi-sumi0)/2)
    sumi1 = sumi-sumi0-sumi2
    outputMT110[idx,:,:] = torch.cat((outputMT11[idx,0:sumi1,:],\
                                  outputMT11[idx,sumi1:sumi-sumi2,:]+outputMT11[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1),:],\
                                  outputMT11[idx+lenn,range(sumi2-1,-1,-1),:], outputMT11[idx,int(sumi):,:]),0)
    targetMT110[idx,:] = torch.cat((targetMT11[idx,0:sumi1], targetMT11[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1)],\
                                  targetMT11[idx+lenn,range(sumi2-1,-1,-1)], targetMT11[idx,int(sumi):]),0)
    targidxMT110[idx,:sumi1+sumi0] = np.append(targidxMT11[idx,0:sumi1], targidxMT11[idx+lenn,range(sumi-sumi1-1,sumi2-1,-1)],0)
    targidxMT110[idx,sumi1+sumi0:] = np.append(targidxMT11[idx+lenn,range(sumi2-1,-1,-1)], targidxMT11[idx,int(sumi):],0)
outputMT110=outputMT110.max(dim=2)[1]
if (targetMT110-targetMT11[0:lenn,:]).max()>0:
    1/0
pred = outputMT110.reshape(-1)[targidxMT110.reshape(-1)>0]
targ = targetMT110.reshape(-1)[targidxMT110.reshape(-1)>0]
perc[0,3] = pred.eq(targ).sum().item()
perc[1,3] = len(targ)
perc[2,3] = pred.eq(targ).sum().item()/len(targ)
