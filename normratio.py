import torch
import torch.nn as nn
import torch.nn.functional as F
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
args = parser.parse_args()

def calculate_layer(model, train_loader, blur_loader, num_blocks, device):
    model.eval()

    norm_pred_ori = dict()
    norm_pred_jigsaw = dict()
    for i in range(num_blocks):
        norm_pred_ori[i] = []
        norm_pred_jigsaw[i] = []

    print(type(model).__name__, len(train_loader), len(blur_loader))
    with torch.no_grad():
        for batch_idx, (data1, data2) in enumerate(zip(train_loader, blur_loader)):
            x = torch.cat([data1[0], data2[0]], dim=0).to(device)

            features = model.forward_features_blockwise(x)  

            for i in range(num_blocks):        
                norm = torch.norm(F.relu(features[i]), dim=[2,3]).mean(1)
                norm_ori = norm[:len(data1[0])]
                norm_jigsaw = norm[len(data1[0]):]

                norm_pred_ori[i].append(norm_ori)
                norm_pred_jigsaw[i].append(norm_jigsaw)

    for i in range(num_blocks):
        norm_pred_ori[i] = torch.cat(norm_pred_ori[i], dim=0)
        norm_pred_jigsaw[i] = torch.cat(norm_pred_jigsaw[i], dim=0)

        print('NormRatio-Block{}: {}'.format(i, (norm_pred_ori[i]/norm_pred_jigsaw[i]).mean())) 

        
def eval():
    config = read_conf('conf/'+args.data+'.json')
    device = 'cuda:'+args.gpu
    dataset_path = config['id_dataset']
    batch_size = config['batch_size']
    save_path = config['save_path'] + args.save_path
    
    num_classes = int(config['num_classes'])

    train_loader, jigsaw_loader = get_cifar_jigsaw(args.data, dataset_path, batch_size)

   
    if  'resnet' in args.net:
        model = resnet.resnet18(num_classes=num_classes)
        num_blocks = 8
    if 'wrn28' == args.net:
        model = wrn.WideResNet(depth=28, widen_factor=10, num_classes=num_classes)
        num_blocks = 12
    if 'vgg11' == args.net:
        model = vgg.VGG(vgg_name = 'VGG11', num_classes = num_classes)
        num_blocks = 8
    model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    model.to(device)
    model.eval()

    calculate_layer(model, train_loader, jigsaw_loader, num_blocks, device)


if __name__ =='__main__':
    eval()