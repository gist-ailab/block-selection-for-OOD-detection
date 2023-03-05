# Block Selection Method for Using Feature Norm in Out-of-distribution Detection (FeatureNorm)
Official Implementation of the **"Block Selection Method for Using Feature Norm in Out-of-distribution Detection (CVPR 2023)"** by Yeonguk Yu, Sungho Shin, Seongju Lee, Changhyun Jun, and Kyoobin Lee. 

In this study, we propose a block selection method that utilizes the L2-norm of the activation map to detect out-of-distribution (OOD) samples. We were inspired to develop this method because we observed that the last block of neural networks can sometimes be overconfident, which can lead to deterioration in OOD detection performance. To select the block for OOD detection, we use NormRatio, which is a ratio of FeatureNorm for ID and pseudo-OOD. This ratio measures the OOD detection performance of each block. Specifically, we create Jigsaw puzzle images from ID training samples to simulate pseudo-OOD and calculate NormRatio. We choose the block with the largest value of NormRatio, which provides the biggest difference between FeatureNorm of ID and FeatureNorm of pseudo-OOD.

![concept.png](/figure/figure_intro.png)



[[ArXiv]](https://arxiv.org/abs/2212.02295)

Currently, this code only supports for the CIFAR10 benchmark with ResNet18 architecture

---
# Updates & TODO Lists
- [x] FeatureNorm has been released
- [ ] pretrained checkpoints
- [x] Environment settings and Train & Evaluation Readme
- [ ] Presentation video
- [ ] Code for all architectures


# Getting Started
## Environment Setup
   This code is tested under Window10 and Python 3.7.7 environment, and the code requires following packages to be installed:
    
   - [Pytorch](https://pytorch.org/): Tested under 1.11.0 version of Pytorch-GPU.
   - [torchvision](https://pytorch.org/vision/stable/index.html): which will be installed along Pytorch. Tested under 0.6.0 version.
   - [timm](https://github.com/rwightman/pytorch-image-models): Tested under 0.4.12 version.
   - [scipy](https://www.scipy.org/): Tested under 1.4.1 version.
   - [scikit-learn](https://scikit-learn.org/stable/): Tested under 0.22.1 version.


## Dataset Preparation
   Some public datasets are required to be downloaded for running evaluation. Required dataset can be downloaded in following links as in https://github.com/wetliu/energy_ood:    
   - [Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/)
   - [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz)
   - [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz)
   - [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz)

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
    
# License
The source code of this repository is released only for academic use. See the [license](LICENSE) file for details.

# Acknowledgement
This work was partially supported by Institute of Information \& communications Technology Planning \& Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00951, Development of Uncertainty-Aware Agents Learning by Asking Questions) and by ICT R\&D program of MSIT/IITP[2020-0-00857, Development of Cloud Robot Intelligence Augmentation, Sharing and Framework Technology to Integrate and Enhance the Intelligence of Multiple Robots].

# Citation
```
@misc{https://doi.org/10.48550/arxiv.2212.02295,
  doi = {10.48550/ARXIV.2212.02295},
  url = {https://arxiv.org/abs/2212.02295},
  author = {Yu, Yeonguk and Shin, Sungho and Lee, Seongju and Jun, Changhyun and Lee, Kyoobin},
  keywords = {Computer Vision and Pattern Recognition (cs.CV), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Block Selection Method for Using Feature Norm in Out-of-distribution Detection},
    publisher = {arXiv},
    year = {2022},  
  copyright = {Creative Commons Attribution 4.0 International}
}

```
