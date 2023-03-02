import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse

from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--net','-n', default = 'resnet18', type=str)
parser.add_argument('--data', '-d', type=str)
parser.add_argument('--gpu', '-g', type=str)
parser.add_argument('--save_path', '-s', type=str)
parser.add_argument('--method' ,'-m', default = 'thnm', type=str)
parser.add_argument('-p3', type=float, default = 3.0)
parser.add_argument('-p4', type=float, default = 1.0)


args = parser.parse_args()

p_pb = args.p3
p_lb = args.p4

def calculate_thnm(model, loader, threshold, device):
    model.eval()
    predictions = []
    w_norm = torch.norm(model.fc.state_dict()['weight'], dim=1, keepdim=True)
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)            
            # ResNet
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.act1(x)
            x = model.maxpool(x)

            x = model.layer1(x)                     
            x = model.layer2(x)                  
            x = model.layer3(x)
            xgap = F.adaptive_avg_pool2d(x, [1,1])
            norm = torch.norm(xgap, p=p_pb, dim=1, keepdim=True)     
            for i in range(len(norm)):
                if norm[i]>threshold[2]:
                    x[i] = x[i]/norm[i] * threshold[2]                        
            x = model.layer4(x) 
            xgap = F.adaptive_avg_pool2d(x, [1,1]).view(-1, x.size(1))
            norm = torch.norm(xgap, p=p_lb, dim=1,keepdim=True)
            predictions.append(norm)
    predictions = torch.cat(predictions).to(device)
    return predictions            
            
def calculate_normthr(model, train_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(train_loader):
            x = inputs.to(device)
            norm = torch.zeros([inputs.size(0), 1]).to(device)
            
            # ResNet
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.act1(x)
            x = model.maxpool(x)

            x = model.layer1(x)                 
            xgap = F.adaptive_avg_pool2d(x, [1,1])
            norm1 = torch.norm(xgap, p=p_pb, dim=1, keepdim=True)       
            x = model.layer2(x)
            xgap = F.adaptive_avg_pool2d(x, [1,1])
            norm2 = torch.norm(xgap, p=p_pb, dim=1, keepdim=True)         
            x = model.layer3(x)
            xgap = F.adaptive_avg_pool2d(x, [1,1])
            norm3 = torch.norm(xgap, p=p_pb, dim=1, keepdim=True)            
            predictions.append(torch.cat([norm1, norm2, norm3], dim=1))
    predictions = torch.cat(predictions, dim=0).to(device)
    return predictions.mean(dim=0).view(-1)

def OOD_results(preds_id, model, loader, thr, device, method, file):  
    #image_norm(loader)
    preds_ood = calculate_thnm(model, loader, thr, device).cpu()

    print(torch.mean(preds_ood), torch.mean(preds_id))
    show_performance(preds_id, preds_ood, method, file=file)
    
def eval():
    config = read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    save_path = config['save_path'] + args.save_path
    
    num_classes = int(config['num_classes'])

    if 'cifar' in args.data:
        train_loader, valid_loader = get_cifar(args.data, dataset_path, batch_size)
    else:
        train_loader, valid_loader = get_train_svhn(dataset_path, batch_size)

   
    if args.net =='resnet18':
        model = timm.create_model(args.net, pretrained=False, num_classes=num_classes)
        model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = torch.nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        n_dims = 512         
    if 'na' in args.save_path:
        model.fc = torch.nn.Linear(n_dims, num_classes, bias=False)
    model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    model.to(device)
    model.eval()

    thr = calculate_normthr(model, train_loader, device)
    print(thr, thr.shape)

    f = open(save_path+'/{}_result.txt'.format(args.method), 'w')
    valid_accuracy = validation_accuracy(model, valid_loader, device)
    f.write('Accuracy for ValidationSet: {}\n'.format(str(valid_accuracy)))

    preds_in = calculate_thnm(model, valid_loader, thr, device).cpu()
    if 'cifar' in args.data:
        OOD_results(preds_in, model, get_svhn(config['svhn'], batch_size), thr, device, args.method+'-SVHN', f)
    else:
        _, cifar_loader = get_cifar('cifar10', config['cifar10'], batch_size)
        OOD_results(preds_in, model, cifar_loader, thr, device, args.method+'-CIFAR10', f)        
    OOD_results(preds_in, model, get_textures(config['textures']), thr, device, args.method+'-TEXTURES', f)
    OOD_results(preds_in, model, get_lsun(config['lsun']), thr, device, args.method+'-LSUN', f)
    OOD_results(preds_in, model, get_lsun(config['lsun-resize']), thr, device, args.method+'-LSUN-resize', f)
    OOD_results(preds_in, model, get_lsun(config['isun']), thr, device, args.method+'-iSUN', f)
    f.close()


if __name__ =='__main__':
    eval()