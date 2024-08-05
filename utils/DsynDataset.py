from torch.utils.data import Dataset


class DsynDataset(Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index], self.target[index]
        return image, label
