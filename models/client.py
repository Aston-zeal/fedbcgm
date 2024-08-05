import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import itertools
from sklearn.metrics import classification_report
import torch.nn.functional as F

def test_img(net_g, datatest, args):
    net_g.eval()
    # testing
    test_loss = 0
    correct = 0
    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)
    y_pred_list = []
    y_true_list = []
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cpu(), target.cpu()
        if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'cinic10':
            resize = transforms.Resize([32,32])
            data = resize(data)
        log_probs = net_g(data)
        # sum up batch loss
        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
        # get the index of the max log-probability
        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()
        y_pred_squeeze = torch.squeeze(y_pred).tolist()
        y_true = target.tolist()
        y_pred_list += y_pred_squeeze
        y_true_list += y_true


    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    # print(classification_report(y_true_list, y_pred_list))
    if args.verbose:
        print('\nTest set: Average loss: {:.4f} \nAccuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss

class MnistDataset(Dataset):
    def __init__(self, dataset, idxs):
        self.idx = idxs
        self.dataset = dataset

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.dataset[self.idx[index]][0], self.dataset[self.idx[index]][1]



class Client(object):
    def __init__(self, args, dataset, idxs, client_id, global_round, dataset_test, dsynDataset=None):
        self.args = args
        self.client_id = client_id  # 客户端id
        self.global_round = global_round  # 当前是第几轮
        self.dataset = dataset
        self.dataset_test = dataset_test
        d = MnistDataset(dataset=dataset, idxs=idxs)
        self.dataloader1 = DataLoader(dataset=d, batch_size=args.local_bs1, shuffle=True) # local train数据集
        self.dataloader2 = None
        if(dsynDataset != None):
            self.dataloader2 = DataLoader(dataset=dsynDataset, batch_size=args.local_bs1, shuffle=True)

            



    def train(self, net, global_weight_collector=None):
        # print(str(self.client_id)+ " Starting Training local model...")
        net.train()
        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.args.lr1, momentum=self.args.momentum)
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-5)
        loss_func = nn.CrossEntropyLoss()
        epoch_loss = []
        # combined_dataloader = itertools.chain(self.dataloader1, self.dataloader2)
        dataloader_list = [self.dataloader1]
        if self.dataloader2 != None:
            dataloader_list.append(self.dataloader2)

        for iter in range(self.args.local_ep1):
            batch_loss = []
            for dataloader in dataloader_list:
                for batch_idx, (images, labels) in enumerate(dataloader):
                    images, labels = images.cpu(), labels.cpu()
                    if self.args.dataset == 'cifar10' or self.args.dataset == 'cifar100' or self.args.dataset == 'cinic10':
                        resize = transforms.Resize([32,32])
                        images = resize(images)

                    net.zero_grad()
                    log_probs = net(images)
                    loss = loss_func(log_probs, labels)

                    if self.args.flmethod == 'fedprox':
                        fed_prox_reg = 0.0
                        for param_index, param in enumerate(net.parameters()):
                            fed_prox_reg += ((1 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                        loss += fed_prox_reg


                    loss.backward()
                    optimizer.step()
                    if self.args.verbose and batch_idx % 10 == 0:
                        print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            iter, batch_idx * len(images), len(self.dataloader1.dataset),
                                   100. * batch_idx / len(self.dataloader1), loss.item()))
                    batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        # acc_test, loss_test = test_img(net, self.dataset_test, self.args)
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)