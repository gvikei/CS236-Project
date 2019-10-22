import os
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch.utils.data import Dataset


class CaptionedImageDataset(Dataset):
    def __getitem__(self, index: int) -> (torch.tensor, str):
        """
        :param index: index of the element to be fetched
        :return: (image : torch.tensor ,caption : str)
        """
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class Tier1ImageNetDataset(CaptionedImageDataset):
    def __init__(self, dirname, max_size=-1):
        """
        :param dirname: path to the npz file to be loaded
        :param max_size: maximum size of the dataset, used for debugging
        """
        self.dirname = dirname
        self.max_size = max_size
        self.classId2className = load_vocab("datasets/tier1/map_clsloc.txt")

        data_files = sorted(os.listdir(dirname))
        self.images = []
        self.labelIds = []

        for i, f in enumerate(data_files):
            print("loading data file {}/{}".format(i + 1, len(data_files)))
            data = np.load(os.path.join(dirname, f))
            self.images.append(data['data'])
            self.labelIds.append(data['labels'])
        self.images = np.concatenate(self.images, axis=0)
        self.labelIds = np.concatenate(self.labelIds)
        self.labelNames = [self.classId2className[y] for y in self.labelIds]

        if max_size >= 0:
            # limit the size of the dataset
            self.labelNames = self.labelNames[:max_size]

    def __getitem__(self, index: int) -> (torch.tensor, str):
        """
        :param index: index of the sample in the dataset
        :return: (image : torch.tensor, caption : str) a 3*32*32 image and the associated caption
        """
        image = torch.tensor(self.images[index]).reshape(3, 32, 32)
        caption = self.labelNames[index].replace("_", " ")
        return (image, caption)

    def __len__(self) -> int:
        return len(self.labelNames)


def load_vocab(vocab_file):
    vocab = {}
    with open(vocab_file) as f:
        for l in f.readlines():
            _, id, name = l[:-1].split(" ")
            vocab[int(id)] = name.replace("_", " ")
    return vocab


if __name__ == "__main__":
    print("Testing dataloader...")
    d = Tier1ImageNetDataset("datasets/tier1/val")
    for i in range(10):
        i = np.random.randint(0, len(d))
        img, text = d[i]
        img = np.transpose(img.reshape((3, 32, 32)), [2, 1, 0])
        plt.figure(figsize=(1.5, 1.5))
        plt.imshow(img)
        plt.title(text)
        plt.show()