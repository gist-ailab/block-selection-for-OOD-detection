import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import argparse

from utils import *
import resnet
import wrn
import vgg

parser = argparse.ArgumentParser()
parser.add_argument('--net','-n', default = 'resnet18', type=str)
parser.add_argument('--data', '-d', type=str)
parser.add_argument('--gpu', '-g', type=str)
parser.add_argument('--save_path', '-s', type=str)
parser.add_argument('--method' ,'-m', default = 'featurenorm', type=str)
args = parser.parse_args()

def calculate_norm(model, loader, device):
    #FeatureNorm from penultimate block
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)         
            # ResNet
            features = model.forward_features_blockwise(x)
            features = features[model.sblock]

            # Norm calculation
            norm = torch.norm(F.relu(features), dim=[2, 3]).mean(1)
            predictions.append(norm)
    predictions = torch.cat(predictions).to(device)
    return predictions            

def calculate_msp(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_idx, (inputs, t) in enumerate(loader):
            x = inputs.to(device)         
            # ResNet
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.act1(x)

            x = model.layer1[0](x)
            x = model.layer1[1](x)                   
            x = model.layer2[0](x)
            x = model.layer2[1](x)
            x = model.layer3[0](x)    
            x = model.layer3[1](x)  
            x = model.layer4[0](x)          
            x = model.layer4[1](x)      
            x = model.global_pool(x).view(-1, 512)
            x = model.fc(x)
            x = torch.softmax(x, dim=1).max(dim=1).values
            predictions.append(x)
    predictions = torch.cat(predictions).to(device)
    return predictions   

if args.method == 'msp':
    calculate_score = calculate_msp
elif args.method == 'featurenorm':
    calculate_score = calculate_norm


def OOD_results(preds_id, model, loader, device, method, file):  
    #image_norm(loader)
    preds_ood = calculate_score(model, loader, device).cpu()

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
        train_loader, valid_loader = get_cifar(args.data, dataset_path, batch_size, eval=True)

    if 'resnet18' == args.net:
        model = resnet.resnet18(num_classes = num_classes)
        model.sblock = 6
    if 'wrn28' == args.net:
        model = wrn.WideResNet(depth=28, widen_factor=10, num_classes=num_classes)
        model.sblock = 10
    if 'vgg11' == args.net:
        model = vgg.VGG(vgg_name = 'VGG11', num_classes = num_classes)
        model.sblock = 5

        
    model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    model.to(device)
    model.eval()

    f = open(save_path+'/{}_result.txt'.format(args.method), 'w')
    valid_accuracy = validation_accuracy(model, valid_loader, device)
    print(valid_accuracy)
    f.write('Accuracy for ValidationSet: {}\n'.format(str(valid_accuracy)))

    preds_in = calculate_score(model, valid_loader, device).cpu()
    OOD_results(preds_in, model, get_svhn('../svhn', batch_size), device, args.method+'-SVHN', f)
    OOD_results(preds_in, model, get_ood('../ood-set/textures/images'), device, args.method+'-TEXTURES', f) # Textures
    OOD_results(preds_in, model, get_ood('../ood-set/LSUN'), device, args.method+'-LSUN-crop', f) # LSUN(c)
    OOD_results(preds_in, model, get_ood('../ood-set/LSUN_resize'), device, args.method+'-LSUN-resize', f) #LSUN(r)
    OOD_results(preds_in, model, get_ood('../ood-set/iSUN'), device, args.method+'-iSUN', f) #iSUN
    OOD_results(preds_in, model, get_places('/SSDd/yyg/data/places256'), device, args.method+'-Places365', f)
    f.close()


if __name__ =='__main__':
    eval()