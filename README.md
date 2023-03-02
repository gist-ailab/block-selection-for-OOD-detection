# Block Selection Method for Using Feature Norm in Out-of-distribution Detection (FeatureNorm)
Official Implementation of the **"Block Selection Method for Using Feature Norm in Out-of-distribution Detection (CVPR 2023)"** by Yeonguk Yu, Sungho Shin, Sungju Lee, Changhyun Jun, and Kyoobin Lee. 
In this study, we propose a block selection method that utilizes the L2-norm of the activation map to detect out-of-distribution (OOD) samples. We were inspired to develop this method because we observed that the last block of neural networks can sometimes be overconfident, which can lead to deterioration in OOD detection performance. To select the block for OOD detection, we use NormRatio, which is a ratio of FeatureNorm for ID and pseudo-OOD. This ratio measures the OOD detection performance of each block. Specifically, we create Jigsaw puzzle images from ID training samples to simulate pseudo-OOD and calculate NormRatio. We choose the block with the largest value of NormRatio, which provides the biggest difference between FeatureNorm of ID and FeatureNorm of pseudo-OOD.

![concept.png](/figure/figure_intro.png)



[[ArXiv]](https://arxiv.org/abs/2212.02295)

# Updates & TODO Lists
- [x] FeatureNorm has been released
- [ ] Demo video and pretrained checkpoints
- [ ] Environment settings and Train & Evaluation Readme
- [ ] Presentation video


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

# Train & Evaluation (not completed)
All networks were trained using a single RTX2080TI GPU (batchsize=128 for CIFAR10)

1. Train Teacher Network (112x112 face images) <br />
    [[Teacher Checkpoint]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/Eg_NHoY_LhxNgUZ4mk3OA-MB_YsE7I3akg6MOoNfEi9yZQ?e=bkJ4z4)
    ```bash
    python train_teacher.py --save_dir $CHECKPOINT_DIR --down_size $DOWN_SIZE --total_iters $TOTAL_ITERS \
                            --batch_size $BATCH_SIZE --gpus $GPU_ID --data_dir $FACE_DIR
    ```

    - You can reference the train scripts in the [$scripts/train_teacher.sh](scripts/train_teacher.sh)
    

2. Train Student Network (14x14, 28x28, 56x56 face images) <br />
    [[Student 14x14]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/EpUj-Qbz9vVKshU2HIVRvjYBLE-rrv-7qUoqUjlrU4pWGg?e=sP5TDp), [[Student 28x28]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/ErwdAAtUceJBgzMShNY7cR8BQzgH1MhO-gg_q1axGc9PIg?e=iArIbK), [[Student 56x56]](https://gisto365-my.sharepoint.com/:f:/g/personal/hogili89_gm_gist_ac_kr/EiSpmbZcNVJMu-uA4OH4qTUBF1oBghvPvTdDAnugjLJmzg?e=u2fFOZ) 
    ```bash
    python train_student.py --save_dir $CHECKPOINT_DIR --down_size $DOWN_SIZE --total_iters $TOTAL_ITERS \
                            --batch_size $BATCH_SIZE --teacher_path $TEACHER_CHECKPOINT_PATH --gpus $GPU_ID \
                            --data_dir $FACE_DIR
    ```
    - You can reference the training scripts in the [$scripts/train_student.sh](scripts/train_student.sh)


3. Evaluation
    ```bash
    python test.py --checkpoint_path $CHECKPOINT_PATH --down_size $DOWN_SIZE --batch_size $BATCH_SIZE --data_dir $FACE_DIR --gpus $GPU_ID
    ```
    
# License
The source code of this repository is released only for academic use. See the [license](LICENSE) file for details.

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
