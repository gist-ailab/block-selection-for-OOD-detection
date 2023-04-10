import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse

from utils import *
import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--net','-n', default = 'resnet18', type=str)
parser.add_argument('--data', '-d', type=str)
parser.add_argument('--gpu', '-g', type=str)
parser.add_argument('--save_path', '-s', type=str)
args = parser.parse_args()

def calculate_layer(model, train_loader, blur_loader, device):
    model.eval()
    norm1_predictions1 = []
    norm1_predictions2 = []

    norm2_predictions1 = []
    norm2_predictions2 = []

    norm3_predictions1 = []
    norm3_predictions2 = []

    norm4_predictions1 = []
    norm4_predictions2 = []  

    norm5_predictions1 = []
    norm5_predictions2 = [] 

    norm6_predictions1 = []
    norm6_predictions2 = [] 

    norm7_predictions1 = []
    norm7_predictions2 = [] 
       
    norm8_predictions1 = []
    norm8_predictions2 = []    

    print(type(model).__name__, len(train_loader), len(blur_loader))
    with torch.no_grad():
        for batch_idx, (data1, data2) in enumerate(zip(train_loader, blur_loader)):
            x = torch.cat([data1[0], data2[0]], dim=0).to(device)

            # ResNet
            if type(model).__name__ == 'ResNet':
                x = model.conv1(x)
                x = model.bn1(x)
                x = model.act1(x)

                x = model.layer1[0](x)
                norm1 = torch.norm(x, dim=[2,3], p=2)
                norm1 = norm1.mean(dim=1)  
                x = model.layer1[1](x)
                norm2 = torch.norm(x, dim=[2,3], p=2)
                norm2 = norm2.mean(dim=1)  
                x = model.layer2[0](x)
                norm3 = torch.norm(x, dim=[2,3], p=2)
                norm3 = norm3.mean(dim=1)       
                x = model.layer2[1](x)
                norm4 = torch.norm(x, dim=[2,3], p=2)
                norm4 = norm4.mean(dim=1)  
                x = model.layer3[0](x)
                norm5 = torch.norm(F.relu(x), dim=[2,3], p=2)
                norm5 = norm5.mean(dim=1)
                x = model.layer3[1](x)
                norm6 = torch.norm(F.relu(x), dim=[2,3], p=2)
                norm6 = norm6.mean(dim=1)
                x = model.layer4[0](x)
                norm7 = torch.norm(F.relu(x), p=2, dim=[2,3])
                norm7 = norm7.mean(dim=1)
                x = model.layer4[1](x)
                norm8 = torch.norm(F.relu(x), p=2, dim=[2,3])
                norm8 = norm8.mean(dim=1)
                

            norm1_predictions1.append(norm1[:len(data1[0])])
            norm1_predictions2.append(norm1[len(data1[0]):])

            norm2_predictions1.append(norm2[:len(data1[0])])
            norm2_predictions2.append(norm2[len(data1[0]):])

            norm3_predictions1.append(norm3[:len(data1[0])])
            norm3_predictions2.append(norm3[len(data1[0]):])

            norm4_predictions1.append(norm4[:len(data1[0])])
            norm4_predictions2.append(norm4[len(data1[0]):])   

            norm5_predictions1.append(norm5[:len(data1[0])])
            norm5_predictions2.append(norm5[len(data1[0]):])  

            norm6_predictions1.append(norm6[:len(data1[0])])
            norm6_predictions2.append(norm6[len(data1[0]):])  

            norm7_predictions1.append(norm7[:len(data1[0])])
            norm7_predictions2.append(norm7[len(data1[0]):])

            norm8_predictions1.append(norm8[:len(data1[0])])
            norm8_predictions2.append(norm8[len(data1[0]):])


    norm1_predictions1 = torch.cat(norm1_predictions1, dim=0)
    norm1_predictions2 = torch.cat(norm1_predictions2, dim=0)

    norm2_predictions1 = torch.cat(norm2_predictions1, dim=0)
    norm2_predictions2 = torch.cat(norm2_predictions2, dim=0)

    norm3_predictions1 = torch.cat(norm3_predictions1, dim=0)
    norm3_predictions2 = torch.cat(norm3_predictions2, dim=0)

    norm4_predictions1 = torch.cat(norm4_predictions1, dim=0)
    norm4_predictions2 = torch.cat(norm4_predictions2, dim=0)

    norm5_predictions1 = torch.cat(norm5_predictions1, dim=0)
    norm5_predictions2 = torch.cat(norm5_predictions2, dim=0)

    norm6_predictions1 = torch.cat(norm6_predictions1, dim=0)
    norm6_predictions2 = torch.cat(norm6_predictions2, dim=0)
    
    norm7_predictions1 = torch.cat(norm7_predictions1, dim=0)
    norm7_predictions2 = torch.cat(norm7_predictions2, dim=0)

    norm8_predictions1 = torch.cat(norm8_predictions1, dim=0)
    norm8_predictions2 = torch.cat(norm8_predictions2, dim=0)
    
    print('norm1: ', (norm1_predictions1/norm1_predictions2).mean())
    print('norm2: ', (norm2_predictions1/norm2_predictions2).mean())
    print('norm3: ', (norm3_predictions1/norm3_predictions2).mean())
    print('norm4: ', (norm4_predictions1/norm4_predictions2).mean())
    print('norm5: ', (norm5_predictions1/norm5_predictions2).mean())
    print('norm6: ', (norm6_predictions1/norm6_predictions2).mean())
    print('norm7: ', (norm7_predictions1/norm7_predictions2).mean())
    print('norm8: ', (norm8_predictions1/norm8_predictions2).mean())

        
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
    model.load_state_dict((torch.load(save_path+'/last.pth.tar', map_location = device)['state_dict']))
    model.to(device)
    model.eval()

    calculate_layer(model, train_loader, jigsaw_loader, device)


if __name__ =='__main__':
    eval()