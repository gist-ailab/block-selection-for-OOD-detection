import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse

from utils import *

def calculate_msp(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)
            outputs = outputs.max(dim=1).values
            predictions.append(outputs)
    predictions = torch.cat(predictions).to(device)
    return predictions

def OOD_results(preds_id, model, loader, device, method, file):  
    preds_ood = calculate_msp(model, loader, device).cpu()

    print(torch.mean(preds_ood), torch.mean(preds_id))
    show_performance(preds_id, preds_ood, method, file=file)


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--net','-n', default = 'resnet18', type=str)
    parser.add_argument('--data', '-d', type=str)
    parser.add_argument('--gpu', '-g', type=str)
    parser.add_argument('--save_path', '-s', type=str)
    parser.add_argument('--method' ,'-m', default = 'msp', type=str)

    args = parser.parse_args()

    config = read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    save_path = config['save_path'] + args.save_path
    
    num_classes = int(config['num_classes'])

    if 'cifar' in args.data:
        _, valid_loader = get_cifar(args.data, dataset_path, batch_size)
    else:
        valid_loader = get_svhn(dataset_path, batch_size)

    if args.net =='resnet18':
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        n_dims = 512


    model.fc = torch.nn.Linear(n_dims, num_classes, bias=False)
    model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    model.to(device)
    model.eval()

    if args.method == 'msp':
        calculate_score = calculate_msp

    f = open(save_path+'/{}_result.txt'.format(args.method), 'w')
    if args.net == 'resnet18':
        valid_accuracy = validation_accuracy(model, valid_loader, device)
        
    f.write('Accuracy for ValidationSet: {}\n'.format(str(valid_accuracy)))
    #MSP
    #image_norm(valid_loader)  
    preds_in = calculate_score(model, valid_loader, device).cpu()
    if 'cifar' in args.data:
        OOD_results(preds_in, model, get_svhn(config['svhn'], batch_size), device, args.method+'-SVHN', f)
    else:
        _, cifar_loader = get_cifar('cifar10', config['cifar10'], batch_size)
        OOD_results(preds_in, model, cifar_loader, device, args.method+'-CIFAR10', f)        
    OOD_results(preds_in, model, get_textures(config['textures']), device, args.method+'-TEXTURES', f)
    OOD_results(preds_in, model, get_lsun(config['lsun']), device, args.method+'-LSUN', f)
    OOD_results(preds_in, model, get_lsun(config['lsun-resize']), device, args.method+'-LSUN-resize', f)
    OOD_results(preds_in, model, get_lsun(config['isun']), device, args.method+'-iSUN', f)
    f.close()

if __name__ =='__main__':
    eval()