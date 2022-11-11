

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
rate_decay = 0.5
kernel_size = [3, 1]
pad_len = int((kernel_size[0]-1)/2)
out_channels = 64

LRI_SIZE = 512
LRI_SIZE1 = 256
HIDDEN_SIZE = 400 
N_LAYER = 2
bidirectional = True #
reverse = False


N_EPOCHS = 400 #


canshu_bian = True
dir = r'.\parameter\modelpara_forward_Q3.pth'
if canshu_bian:
    start_epoch = 1
else:
    checkpoint = torch.load(dir)
    start_epoch = checkpoint['epoch'] + 1

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
        x = x.permute(0, 2, 3, 1).reshape(batch_size, -1, out_channels)
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

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def trainModel(epoch):
    # 将training =True
    CRNNmodel.train()
    total_loss = 0
    for i, data in enumerate(trainloader, 1):
        # inputs (batch_size, seq_len, INPUTS_SIZE)
        # target (batch_size, seq_len, LABELS_NUM)
        if reverse:
            inputs, _, _, target = make_tensors1(data)
        else:
            inputs, _, _, target = make_tensors(data)
        batch_size = inputs.size(0)
        output, _ = CRNNmodel(inputs)
        loss = criterion(output.reshape(-1, LABELS_NUM), target.reshape(-1, LABELS_NUM)) * 1000
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}] ', end='')
            print(f'loss={total_loss / (i * len(inputs))}')
    return total_loss


def valiModel():
    # training =False
    CRNNmodel.eval()
    correct = 0
    total = 0
    print("validating trained model ...")
    # 表示不需要求梯度
    with torch.no_grad():
        for i, data in enumerate(valiloader, 1):
            # inputs (batch_size, seq_len, INPUTS_SIZE)
            # target (batch_size, seq_len, LABELS_NUM)
            if reverse:
                inputs, targlc, targidx, _ = make_tensors1(data)
            else:
                inputs, targlc, targidx, _ = make_tensors(data)
            output, _ = CRNNmodel(inputs)
            output = output.reshape(-1, LABELS_NUM)
            pred = output.max(dim=1)[1][targidx].cpu()
            targ = targlc.cpu()[targidx]
            correct += pred.eq(targ.view_as(pred)).sum().item()
            total += len(targ)

        percent = '%.2f' % (100 * correct / total)
        print(f'Validation set: Accuracy {correct}/{total} {percent}%')

    return correct / total


def testModel():
    # training =False
    CRNNmodel.eval()
    correct = 0
    total = 0
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, data in enumerate(testloader, 1):
            # inputs (batch_size, seq_len, INPUTS_SIZE)
            # target (batch_size, seq_len, LABELS_NUM)
            if reverse:
                inputs, targlc, targidx, _ = make_tensors1(data)
            else:
                inputs, targlc, targidx, _ = make_tensors(data)
            output, _ = CRNNmodel(inputs)

            output = output.reshape(-1, LABELS_NUM)
            pred = output.max(dim=1)[1][targidx].cpu()
            targ = targlc.cpu()[targidx]

            correct += pred.eq(targ.view_as(pred)).sum().item()
            total += len(targ)

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct / total, list(pred), list(targ)

def testModel513():
    # training =False
    CRNNmodel.eval()
    correct = 0
    total = 0
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, data in enumerate(testloader513, 1):
            # inputs (batch_size, seq_len, INPUTS_SIZE)
            # target (batch_size, seq_len, LABELS_NUM)
            if reverse:
                inputs, targlc, targidx, _ = make_tensors1(data)
            else:
                inputs, targlc, targidx, _ = make_tensors(data)
            # 1 forward算模型输出，2forward算loss 3.梯度清零 4.反向传播求梯度 5.更新
            output, _ = CRNNmodel(inputs)

            output = output.reshape(-1, LABELS_NUM)
            pred = output.max(dim=1)[1][targidx].cpu()
            targ = targlc.cpu()[targidx]

            correct += pred.eq(targ.view_as(pred)).sum().item()
            total += len(targ)

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set513: Accuracy {correct}/{total} {percent}%')

    return correct / total, list(pred), list(targ)


def testModelCASP10():
    # training =False
    CRNNmodel.eval()
    correct = 0
    total = 0
    print("evaluating trained model ...")
    # 表示不需要求梯度
    with torch.no_grad():
        for i, data in enumerate(testloaderCASP10, 1):
            # inputs (batch_size, seq_len, INPUTS_SIZE)
            # target (batch_size, seq_len, LABELS_NUM)
            if reverse:
                inputs, targlc, targidx, _ = make_tensors1(data)
            else:
                inputs, targlc, targidx, _ = make_tensors(data)
            output, _ = CRNNmodel(inputs.float())
            output = output.reshape(-1, LABELS_NUM)
            pred = output.max(dim=1)[1][targidx].cpu()
            targ = targlc.cpu()[targidx]
            correct += pred.eq(targ).sum().item()
            total += len(targ)
        percent = '%.2f' % (100 * correct / total)
        print(f'Test setCASP10: Accuracy {correct}/{total} {percent}%')

    return correct / total


def testModelCASP11():
    # training =False
    CRNNmodel.eval()
    correct = 0
    total = 0
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, data in enumerate(testloaderCASP11, 1):
            # inputs (batch_size, seq_len, INPUTS_SIZE)
            # target (batch_size, seq_len, LABELS_NUM)
            if reverse:
                inputs, targlc, targidx, _ = make_tensors1(data)
            else:
                inputs, targlc, targidx, _ = make_tensors(data)
            output, _ = CRNNmodel(inputs.float())
            output = output.reshape(-1, LABELS_NUM)
            pred = output.max(dim=1)[1][targidx].cpu()
            targ = targlc.cpu()[targidx]
            correct += pred.eq(targ).sum().item()
            total += len(targ)
        percent = '%.2f' % (100 * correct / total)
        print(f'Test setCASP11: Accuracy {correct}/{total} {percent}%')

    return correct / total


CRNNmodel = CRNNmodel()
if canshu_bian:
    start_epoch = 1
else:
    CRNNmodel.load_state_dict(checkpoint['net'])
    minacc = checkpoint['minacc']

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CRNNmodel.parameters(), lr=learning_rate, weight_decay=LAM)  # 优化器
scheduler = StepLR(optimizer, step_size=1, gamma=rate_decay)

start = time.time()


if USE_GPU:
    device = torch.device("cuda:0")
    CRNNmodel.to(device)
    criterion.to(device)

print("Training for %d epochs..." % N_EPOCHS)
acc_list = []
acc_list = list(acc_list)
valiacc_list = []
valiacc_list = list(valiacc_list)
for epoch in range(start_epoch, N_EPOCHS + 1):
    # Train cycle
    trainModel(epoch)
    valiacc = valiModel()
    valiacc_list.append(valiacc)
    if epoch == 1: minacc = valiacc
    if np.array(valiacc_list)[-1] < minacc:
        # break
        print("第%d个epoch的学习率：%f" % (epoch, optimizer.param_groups[0]['lr']))
        scheduler.step()
        checkpoint = torch.load(dir)
        CRNNmodel.load_state_dict(checkpoint['net'])
    else:
        minacc = np.array(valiacc_list)[-1]
        state = {'net': CRNNmodel.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'minacc': minacc}
        torch.save(state, dir)
    if optimizer.param_groups[0]['lr'] < 0.0001:
        break

checkpoint = torch.load(dir)
CRNNmodel.load_state_dict(checkpoint['net'])
acc, lsp, lst = testModel()
acc1, lsp1, lst1 = testModel513()
acc2 = testModelCASP10()
acc3 = testModelCASP11()