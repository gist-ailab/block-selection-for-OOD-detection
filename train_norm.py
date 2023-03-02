import os
import torch
import torchvision
import argparse
import utils
import timm
import numpy as np

import torch.nn.functional as F
import torch.nn as nn

    
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--weight', '-w', default = 1.0, type=float)

    args = parser.parse_args()

    config = utils.read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    
    model_name = args.net
    dataset_path = config['id_dataset']
    save_path = config['save_path'] + args.save_path
    num_classes = int(config['num_classes'])
    class_range = list(range(0, num_classes))

    if args.net == 'resnet18':
        batch_size = int(config['batch_size'])
        max_epoch = int(config['epoch'])
        wd = 5e-04
        lrde = [50, 75, 90]

    if args.data == 'svhn':
        wd = 1e-04

    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
        
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    if args.data == 'cifar10' or args.data=='cifar100':
        train_loader, valid_loader = utils.get_cifar(args.data, dataset_path, batch_size)
    else:
        train_loader, valid_loader = utils.get_train_svhn(dataset_path, batch_size)

    if 'resnet' in args.net:
        def norm_forward_features(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.act1(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            if self.training:
                feature = F.adaptive_avg_pool2d(x, [1,1])
                norm = torch.norm(feature, dim=1, keepdim=True)
                x = x/norm
            return x

        timm.models.resnet.ResNet.forward_features = norm_forward_features
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        model.fc = torch.nn.Linear(512, num_classes, bias=False)   
        
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.1, momentum=0.9, weight_decay = wd)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)   

    logit_weight = np.log(99 * num_classes - 99) * 1.0
    epoch_weight = 1/(max_epoch)
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0 
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            features = model.forward_features(inputs)
            features = model.global_pool(features)
            
            outputs = model.fc(features)

            outputs = outputs * logit_weight * (1.0+(epoch_weight*(max_epoch - epoch)))
                 
            loss = criterion(outputs, targets)
            
            loss.backward()            
            optimizer.step()
            
            total_loss += loss
            total += targets.size(0)
            _, predicted = outputs.max(1)            
            correct += predicted.eq(targets).sum().item()      
            print('\r', batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                        % (total_loss/(batch_idx+1), 100.*correct/total, correct, total), end = '')                       
        train_accuracy = correct/total
        train_avg_loss = total_loss/len(train_loader)
        print()

        ## validation
        model.eval()
        total_loss = 0
        total = 0
        correct = 0
        valid_accuracy = utils.validation_accuracy(model, valid_loader, device)

        scheduler.step()

        saver.save_checkpoint(epoch, metric = valid_accuracy)
        print('EPOCH {:4}, TRAIN [loss - {:.4f}, acc - {:.4f}], VALID [acc - {:.4f}]\n'.format(epoch, train_avg_loss, train_accuracy, valid_accuracy))
        print(scheduler.get_last_lr())

if __name__ =='__main__':
    train()