import numpy as np
import collections
from scipy.spatial import distance
from sklearn.metrics import jaccard_score
from torchvision import datasets, transforms
import torch
from utils.DsynDataset import DsynDataset
from sklearn.mixture import GaussianMixture
import sys


class Databalance:
    def __init__(self, size_device, dataset, alpha, frac, gama=6):
        # data load from data set
        self.train_data = None
        self.test_data = None
        self.dict_users = None

        self.dataset = dataset
        self.alpha = alpha

        self.size_device = size_device
        self.size_class = None
        self.mediator = []
        self.gamma = 6

        self.idx_users = np.random.choice(range(self.size_device), int(self.size_device * frac),
                                     replace=False).tolist()

        # restore DsynDataset
        self.DsynDataset_dict = {}

    def get_bc(self, input1, input2):
        # c1, c2 = collections.Counter(input1), collections.Counter(input2)
        c2 = collections.Counter(input2)
        # num_elements = len(c2)
        # print("Number of elements in c2:", num_elements)
        c1 = {}
        if self.dataset == 'fashionmnist' or self.dataset == 'emnist':
            for i in range(self.size_class):
                c1[i] = torch.sum(torch.eq(self.train_data.targets, i))
        else:
            c1 = collections.Counter(input1)
        d1, d2 = [], []
        for key in c1.keys():
            d1.append(c1[key] / len(input1))
            d2.append(c2[key] / len(input2))
        return distance.braycurtis(d1, d2)
        # intersection = sum(min(d1_val, d2_val) for d1_val, d2_val in zip(d1, d2))
        # union = sum(max(d1_val, d2_val) for d1_val, d2_val in zip(d1, d2))
        # jaccard_index = intersection / union
        # return 1 - jaccard_index

    def get_client_targets(self, client):
        client_targets = []
        client_idxs = self.dict_users[client]
        for idx in client_idxs:
            client_targets.append(self.train_data[idx][1])
        # print(client_targets)
        return client_targets

    def get_client_feature(self, client):
        client_feature = []
        client_idxs = self.dict_users[client]
        for idx in client_idxs:
            client_feature.append(self.train_data[idx][0])
        return client_feature


    def assign_clients(self, balance = True):
        # idx_users = np.random.choice(range(self.size_device), int(self.size_device * 0.4),
        #                              replace=False).tolist()
        print("idx_users:",self.idx_users)
        if not balance:
            self.mediator = [{i} for i in self.idx_users]
            return
        client_pool = set([i for i in self.idx_users])
        while client_pool:
            new_mediator = set()
            mediator_label_pool = np.array([])
            while client_pool and len(new_mediator) < self.gamma:
                select_client, kl_score = None, float('inf')
                for client in client_pool:
                    mediator_label_pool_tmp = np.hstack([mediator_label_pool, self.get_client_targets(client)])
                    new_kl_score = self.get_bc(self.train_data.targets, mediator_label_pool_tmp)
                    print("new_bc:{}, client:{}, bc:{}".format(new_kl_score, client, kl_score))
                    if new_kl_score < kl_score:
                        select_client = client
                        kl_score = new_kl_score
                new_mediator.add(select_client)
                mediator_label_pool = np.hstack([mediator_label_pool, self.get_client_targets(select_client)])
                client_pool.remove(select_client)
            self.mediator.append(new_mediator)
        print(self.mediator)

    def get_dataset(self):
        print("=" * 20, self.dataset, "=" * 20)
        if self.dataset == 'mnist':
            trans_mnist = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            dataset_train = datasets.MNIST(
                '../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST(
                '../data/mnist/', train=False, download=True, transform=trans_mnist)
        elif self.dataset == 'fashionmnist':
            trains_femnist = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3801,))
            ])
            dataset_train = datasets.FashionMNIST(
                '../data/fashionmnist/', train=True, download=True, transform=trains_femnist)
            dataset_test = datasets.FashionMNIST(
                '../data/fashionmnist/', train=False, download=True, transform=trains_femnist)
        elif self.dataset == 'emnist':
            trains_femnist = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            dataset_train = datasets.EMNIST(
                '../data/emnist/', train=True, download=True, transform=trains_femnist, split="byclass")
            dataset_test = datasets.EMNIST(
                '../data/emnist/', train=False, download=True, transform=trains_femnist, split="byclass")
        elif self.dataset == 'cifar10':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(64), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        elif self.dataset == 'cifar100':
            trans_cifar = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(64), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
            dataset_train = datasets.CIFAR100('../data/cifar100', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR100('../data/cifar100', train=False, download=True, transform=trans_cifar)
        elif self.dataset == 'cinic10':
            trans_cinic10 = transforms.Compose(
                [transforms.ToTensor(), transforms.Resize(64), transforms.Normalize((0.47889522, 0.47227842, 0.43047404), (0.24205776, 0.23828046, 0.25874835))])
            dataset_train = datasets.ImageFolder('../data/cinic10', transform=trans_cinic10)
            dataset_test = datasets.ImageFolder('../data/cinic10', transform=trans_cinic10)
        print(len(dataset_train))
        print(len(dataset_test))
        self.train_data = dataset_train
        # print("torch.sum(torch.eq(self.train_data.targets, 0))=======", torch.sum(torch.eq(self.train_data.targets, 0)))
        self.test_data = dataset_test
        self.size_class = len(dataset_train.classes)


    def dirichlet_split_noniid(self):
        print(self.alpha)
        ori_dataset = [self.train_data]
        NUM_CLASS = len(self.train_data.classes)
        MIN_SIZE = 0
        X = [[] for _ in range(self.size_device)]
        Y = [[] for _ in range(self.size_device)]
        stats = {}
        targets_numpy = np.concatenate(
            [ds.targets for ds in ori_dataset], axis=0, dtype=np.int64
        )
        data_numpy = np.concatenate(
            [ds.data for ds in ori_dataset], axis=0, dtype=np.float32
        )
        print("data_numpy.shape=======",data_numpy.shape)
        idx = [np.where(targets_numpy == i)[0] for i in range(NUM_CLASS)]
        dict_users = {i: np.array([], dtype='int64') for i in range(self.size_device)}
        while MIN_SIZE < 10:
            idx_batch = [[] for _ in range(self.size_device)]
            for k in range(NUM_CLASS):
                np.random.shuffle(idx[k])
                distributions = np.random.dirichlet(np.repeat(self.alpha, self.size_device))
                distributions = np.array(
                    [
                        p * (len(idx_j) < len(targets_numpy) / self.size_device)
                        for p, idx_j in zip(distributions, idx_batch)
                    ]
                )
                distributions = distributions / distributions.sum()
                distributions = (np.cumsum(distributions) * len(idx[k])).astype(int)[:-1]
                idx_batch = [
                    np.concatenate((idx_j, idx.tolist())).astype(np.int64)
                    for idx_j, idx in zip(idx_batch, np.split(idx[k], distributions))
                ]

                MIN_SIZE = min([len(idx_j) for idx_j in idx_batch])

            for i in range(self.size_device):
                stats[i] = {"x": None, "y": None}
                np.random.shuffle(idx_batch[i])
                X[i] = data_numpy[idx_batch[i]]
                Y[i] = targets_numpy[idx_batch[i]]
                stats[i]["x"] = len(X[i])
                stats[i]["y"] = collections.Counter(Y[i].tolist())
                dict_users[i] = idx_batch[i]

        self.dict_users = dict_users

    def create_virtual_data_v3(self):
        # 1.统计每个本地客户端各个类对应样本的均值、方差、数量
        print("1.统计每个本地客户端各个类对应样本的均值、方差、数量")
        label_to_mean_dict_global = {}
        label_to_num_dict_global = {}
        label_to_var_dict_global = {}
        for k in range(self.size_device):
            num_each_label = {}
            var_each_label = {}
            mean_each_label = {}

            for i in range(self.size_class):
                feature = self.get_client_feature(k)
                feature_np = np.array(feature)
                mean = np.mean(feature_np, axis=0)
                var = np.var(feature_np, axis=0)
                mean_each_label[i] = mean
                var_each_label[i] = var
                num_each_label[i] = self.get_client_targets(k).count(i)

            # 2.传本地客户端各个类对应样本的均值、方差、数量到服务器
            print("2.传本地客户端各个类对应样本的均值、方差、数量到服务器")
            label_to_mean_dict_global[k] = mean_each_label
            label_to_num_dict_global[k] = num_each_label
            label_to_var_dict_global[k] = var_each_label
            num1 = sys.getsizeof(label_to_mean_dict_global)
            num2 = sys.getsizeof(label_to_num_dict_global)
            num3 = sys.getsizeof(label_to_var_dict_global)
            # print("均值的通信开销：{}".format(num1))
            # print("方差的通信开销：{}".format(num3))
            # print("分布的通信开销：{}".format(num2))

        # 3.服务器以数量作为权重聚合本地客户端各个类的均值、方差，生成全局数据的均值、方差
        print("3.服务器以数量作为权重聚合本地客户端各个类的均值、方差，生成全局数据的均值、方差")
        global_mean = {}
        global_var = {}
        num_each_class_global = np.zeros(self.size_class)
        for i in range(self.size_class):
            if self.dataset == 'fashionmnist' or self.dataset == 'emnist':
                num_each_class_global[i] = torch.sum(torch.eq(self.train_data.targets, i))
            else:
                num_each_class_global[i] = self.train_data.targets.count(i)
        for k in range(self.size_device):
            for key, value in label_to_num_dict_global[k].items():
                label_to_num_dict_global[k][key] = value / num_each_class_global[key]
        for k in range(self.size_device):
            for key, value in label_to_mean_dict_global[k].items():
                if key in global_mean:
                    global_mean[key] += label_to_num_dict_global[k][key] * label_to_mean_dict_global[k][key]
                    global_var[key] += label_to_num_dict_global[k][key] * label_to_var_dict_global[k][key]
                else:
                    global_mean[key] = label_to_num_dict_global[k][key] * label_to_mean_dict_global[k][key]
                    global_var[key] = label_to_num_dict_global[k][key] * label_to_var_dict_global[k][key]
        for k in range(self.size_device):
            for key, value in label_to_num_dict_global[k].items():
                label_to_num_dict_global[k][key] = value * num_each_class_global[key]
        # 4.使用高斯混合模型生成仿真数据，使每组各个类别的样本数一致
        print("# 4.使用高斯混合模型生成仿真数据，使每组各个类别的样本数一致")
        for index, group in enumerate(self.mediator):
            print("Group {:2d} begin produce virtual data".format(index))
            total_num_each_class_group = {}
            for device in group:
                for key, value in label_to_num_dict_global[device].items():
                    if key not in total_num_each_class_group:
                        total_num_each_class_group[key] = value
                    else:
                        total_num_each_class_group[key] += value

            values = total_num_each_class_group.values()
            max_value = sum(values) / len(values)
            max_value = int(max_value)

            Dsyn_data = []
            Dsyn_label = []
            for label, value in total_num_each_class_group.items():
                if (value < max_value):
                    num = max_value - value
                    num = int(num)
                    print('Data about Label {:3d} start produce, number is {:6d}'.format(label, num))
                    for i in range(num):
                        virtual_data = self.gmm(global_mean[label], global_var[label])
                        Dsyn_data.append(virtual_data)
                        Dsyn_label.append(label)
            dsynDataset = DsynDataset(Dsyn_data, Dsyn_label)
            self.DsynDataset_dict[index] = dsynDataset


    def gmm(self, mean, var):
        mean = torch.from_numpy(mean)
        var = torch.from_numpy(var)
        virtual_data = torch.randn(mean.shape).to("cpu") * \
                       torch.sqrt(var).to("cpu") + \
                       mean.to("cpu")
        return virtual_data

    def split_dirichlet(self):

        label_distribution = np.random.dirichlet([self.alpha] * self.size_device, self.size_class)
        label_distribution = self.make_double_stochstic(label_distribution)

        ori_dataset = [self.train_data]
        targets_numpy = np.concatenate(
            [ds.targets for ds in ori_dataset], axis=0, dtype=np.int64
        )
        class_idcs = [np.where(targets_numpy == i)[0] for i in range(self.size_class)]
        client_idcs = [[] for _ in range(self.size_device)]
        for c, fracs in zip(class_idcs, label_distribution):
            for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
                client_idcs[i] += [idcs]

        client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
        self.dict_users = client_idcs

    def make_double_stochstic(self, x):
        rsum = None
        csum = None
        n = 0
        while n < 1000 and (np.any(rsum != 1) or np.any(csum != 1)):
            x /= x.sum(0)
            x = x / x.sum(1)[:, np.newaxis]
            rsum = x.sum(1)
            csum = x.sum(0)
            n += 1
        return x











