import json
import torch.utils.data as data
import torch
import random

import torchvision
from torchvision import transforms
from torchvision import datasets as dset


size = 32
train_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.RandomHorizontalFlip(), transforms.RandomCrop(size, padding=4),
                               transforms.ToTensor()])
test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor()])#, )


class jigsaw_dataset(data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        x, y = self.dataset[index]
        
        s = int(float(x.size(1)) / 3)
        
        
        x_ = torch.zeros_like(x)
        tiles_order = random.sample(range(9), 9)
        for o, tile in enumerate(tiles_order):
            i = int(o/3)
            j = int(o%3)
            
            ti = int(tile/3)
            tj = int(tile%3)
            # print(i, j, ti, tj)
            x_[:, i*s:(i+1)*s, j*s:(j+1)*s] = x[:, ti*s:(ti+1)*s, tj*s:(tj+1)*s] 
        return x_, y
        
def get_cifar_jigsaw(dataset, folder, batch_size, test=False):
    test_transform_cifar = transforms.Compose([transforms.Resize([size,size]), transforms.ToTensor()])
    train_ = not test
    
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=train_, transform=test_transform_cifar, download=True)
        test_data = dset.CIFAR10(folder, train=train_, transform=test_transform_cifar, download=True)

    jigsaw = jigsaw_dataset(test_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    jigsaw_loader = torch.utils.data.DataLoader(jigsaw, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, jigsaw_loader

def read_conf(json_path):
    """
    read json and return the configure as dictionary.
    """
    with open(json_path) as json_file:
        config = json.load(json_file)
    return config
    
def get_cifar(dataset, folder, batch_size, eval=False):
    if eval==True:
        train_transform_cifar_ = test_transform_cifar
    else:
        train_transform_cifar_ = train_transform_cifar
    if dataset == 'cifar10':
        train_data = dset.CIFAR10(folder, train=True, transform=train_transform_cifar_, download=True)
        test_data = dset.CIFAR10(folder, train=False, transform=test_transform_cifar, download=True)
        num_classes = 10
        
    train_loader = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True, pin_memory=True, num_workers = 4)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)
    
    return train_loader, valid_loader

def get_svhn(folder, batch_size, transform_imagenet = False):
    test_data = dset.SVHN(folder, split='test', transform=test_transforms if transform_imagenet else test_transform_cifar, download=True)
    valid_loader = torch.utils.data.DataLoader(test_data, batch_size, shuffle=False, pin_memory=True, num_workers = 4)    
    return valid_loader


def get_ood(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)
    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader    


def get_places(path):
    ood_data = torchvision.datasets.ImageFolder(path, test_transform_cifar)

    random.seed(0)
    ood_data.samples = random.sample(ood_data.samples, 10000)

    ood_loader = torch.utils.data.DataLoader(ood_data, batch_size=100, shuffle=False, pin_memory=True)
    return ood_loader

if __name__ == '__main__':
    pass