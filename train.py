import os
import torch
import torchvision
import argparse
import numpy as np
import timm

import utils
import resnet
import wrn
import vgg

torch.manual_seed(0)   

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', default = '0', type=str)
    parser.add_argument('--save_path', '-s', type=str)

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
        lr = 0.1
    if args.net == 'wrn28':
        batch_size = int(config['batch_size'])
        max_epoch = 200
        wd = 5e-04
        lrde = [100, 150]
        lr = 0.1
    if args.net == 'vgg11':
        batch_size = int(config['batch_size'])
        max_epoch = int(config['epoch'])
        wd = 5e-04
        lrde = [50, 75, 90]
        lr = 0.05

    print(model_name, dataset_path.split('/')[-2], batch_size, class_range)
    
    if not os.path.exists(config['save_path']):
        os.mkdir(config['save_path'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    else:
        raise ValueError('save_path already exists')
    
    if 'cifar' in args.data:
        train_loader, valid_loader = utils.get_cifar(args.data, dataset_path, batch_size)

    if 'resnet18' == args.net:
        model = resnet.resnet18(num_classes = num_classes)
    if 'wrn28' == args.net:
        model = wrn.WideResNet(depth=28, widen_factor=10, num_classes=num_classes)
    if 'vgg11' == args.net:
        model = vgg.VGG(vgg_name = 'VGG11', num_classes = num_classes)
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()    
    optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.9, weight_decay = wd)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, lrde)
    saver = timm.utils.CheckpointSaver(model, optimizer, checkpoint_dir= save_path, max_history = 2)    
    
    for epoch in range(max_epoch):
        ## training
        model.train()
        total_loss = 0
        total = 0
        correct = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
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