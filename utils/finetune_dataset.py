from torch.utils.data import DataLoader, Dataset
class MyDataset(Dataset):
    def __init__(self, dataset, num_per_generator):
        self.dataset = dataset
        self.num = num_per_generator

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index], int(index/self.num)