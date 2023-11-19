import torch
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self, pics, labels):
        self.pics = pics
        self.labels = labels

    def __getitem__(self, index):
        assert index < len(self.pics)
        return torch.Tensor(self.pics[index]), self.labels[index]

    def __len__(self):
        return len(self.pics)

    def get_tensors(self):
        return torch.Tensor([self.pics]), torch.Tensor(self.labels)


if __name__ == '__main__':
    pass
