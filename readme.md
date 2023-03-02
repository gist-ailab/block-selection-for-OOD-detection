# FeatureNorm

This is implementation of the paper entitled "Norm-based Out-of-distribution Detection".

![concept.png](/figure/figure_intro.png)

## Preliminaries
This code is tested under Window10 and Python 3.7.7 environment, and this code requires following packages to be installed:

- [Pytorch](https://pytorch.org/): Tested under 1.11.0 version of Pytorch-GPU.
- [torchvision](https://pytorch.org/vision/stable/index.html): which will be installed along Pytorch. Tested under 0.6.0 version.
- [timm](https://github.com/rwightman/pytorch-image-models): Tested under 0.4.12 version.
- [scipy](https://www.scipy.org/): Tested under 1.4.1 version.
- [scikit-learn](https://scikit-learn.org/stable/): Tested under 0.22.1 version.


## Download Dataset
Some public datasets are required to be downloaded for running evaluation. Required dataset can be downloaded in following links as in https://github.com/wetliu/energy_ood:
- [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
- [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
- [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
- [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)



## How to Run

### Config file need to be changed for your path to download. For example,
~~~
# conf/cifar10.json
{
    "epoch" : "100",
    "id_dataset" : "./cifar10",   # Your path to Cifar10
    "batch_size" : 128,
    "save_path" : "./cifar10/",   # Your path to checkpoint
    "num_classes" : 10,
    "svhn": "./svhn",  # Your path to SVHN
    "textures": "./textures", # Your path to Textures
    "lsun": "./lsun", # Your path to LSUN-C
    "lsun-resize": "./lsun_resized", # Your path to LSUN-R
    "isun": "./isun" # Your path to iSUN
}
~~~

### To train a model by our setting (i.e., ours) with ResNet18 architecture
~~~
python train_norm.py -d 'data_name' -g 'gpu_num' -s 'save_name'
~~~
for example, 
~~~
python train_norm.py -d cifar10 -g 0 -s norm_network
~~~

- - -
### To evaluate a model on OOD benchmark using MSP
~~~
python eval.py -d 'data_name' -g 'gpu_num' -s 'model_name'
~~~
for example, 
~~~
python eval.py -d cifar10 -g 0 -s norm_network   #for evaluating our model with MSP detection method
~~~
---
### To evaluate a model on OOD benchmark using FeatureNorm
~~~
python eval_norm.py -d 'data_name' -g 'gpu_num' -s 'model_name'
~~~
for example, 
~~~
python eval_norm.py -d cifar10 -g 0 -s norm_network   #for evaluating our model with FeatureNorm method
~~~
