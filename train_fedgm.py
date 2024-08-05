import time

from utils.options import args_parser
from utils.Databalance import Databalance
from models.Nets import MLP, ConvNet, LargeFNN
from models.test import test_img
from models.client import Client
from models.Fed import FedAvg
import numpy as np
import copy
import matplotlib.pyplot as plt

def fedgm_train(args, db, net_glob):
    w_glob = net_glob.state_dict()
    loss_train = []
    cv_loss, cv_acc = [], []

    loss_test = []
    acc_test = []

    global_weight_collector = list(net_glob.to('cpu').parameters())

    for epoch in range(args.epochs):
        loss_locals = []
        w_locals = []
        start_time = time.time()

        for i, mdt in enumerate(db.mediator):
            # if len(mdt) != db.gamma:
            #     continue
            # print('Group {:3d} start local train'.format(i))
            need_index = [db.dict_users[k] for k in mdt]
            dsynDataset = db.DsynDataset_dict[i] if len(db.DsynDataset_dict) > 0 else None
            local = Client(args=args, dataset=db.train_data, idxs=np.hstack(need_index), client_id=i,
                           global_round=epoch, dataset_test=db.test_data, dsynDataset=dsynDataset)
            if args.flmethod == 'fedprox':
                w, loss = local.train(net=copy.deepcopy(net_glob).cpu(), global_weight_collector = global_weight_collector)
            else:
                w, loss = local.train(net=copy.deepcopy(net_glob).cpu())
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        # ----- fedavg聚合 -----
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(epoch, loss_avg))
        loss_train.append(loss_avg)

        net_glob.eval()
        res = get_acc(net_glob, db.test_data, args)
        net_glob.train()
        end_time = time.time()
        total_time = end_time - start_time
        print("one global epoch need time:{}s".format(total_time))

    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig(
        './testoutput/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))



def get_acc(net_glob, dataset_test, args):
    # acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    # print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))
    return acc_test, loss_test

def get_model(args):
    if args.dataset == 'mnist' or args.dataset == 'fashionmnist' or args.dataset == 'emnist':
        # net_glob = CNNMnist(args=args).cpu()
        num_classes = 62 if args.dataset == 'emnist' else 10
        net_glob = MLP(num_classes=num_classes).cpu()
        if args.model == 'ConvNet':
            net_glob = ConvNet(num_classes=num_classes, dataset = 'mnist').cpu()
    elif args.dataset == 'cifar10':
        # net_glob = CNNCifar(args=args).cpu()
        net_glob = ConvNet(num_classes=10).cpu()
    elif args.dataset == 'cifar100':
        # net_glob = CNNCifar(args=args).cpu()
        net_glob = ConvNet(num_classes=100).cpu()
    elif args.dataset == 'cinic10':
        # net_glob = CNNCifar(args=args).cpu()
        net_glob = ConvNet(num_classes=10).cpu()
    return net_glob

def flatten(source):
    return torch.cat([value.flatten() for value in source.values()])

if __name__ == '__main__':
    args = args_parser()
    args.local_ep1 = 2
    db = Databalance(args.num_users, args.dataset, args.alpha, args.frac)
    db.get_dataset()
    db.split_dirichlet()

    net_glob = get_model(args)
    net_glob.train()

    if args.flmethod == 'fedprox':
        db_avg = copy.deepcopy(db)
        db_avg.assign_clients(False)
        fedgm_train(args, db_avg, net_glob)
    elif args.flmethod == 'fedBCGM_group':
        db_fed_group = copy.deepcopy(db)
        db_fed_group.assign_clients(True)
        fedgm_train(args, db_fed_group, net_glob)
    elif args.flmethod == 'fedBCGM':
        db_fedgm = copy.deepcopy(db)
        db_fedgm.assign_clients(True)
        db_fedgm.create_virtual_data_v3()
        fedgm_train(args, db_fedgm, net_glob)
    else: #fedavg
        db_avg = copy.deepcopy(db)
        db_avg.assign_clients(False)
        fedgm_train(args, db_avg, net_glob)






