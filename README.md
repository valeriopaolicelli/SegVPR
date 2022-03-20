# SegVPR
![Learning Semantics for Visual Place Recognition through Multi-Scale Attention](https://arxiv.org/pdf/2201.09701.pdf) accepted @ICIAP2022  
![Architecture](images/architecture.jpg?raw=true)  
![MS-Attention-Module](images/ms_attention_module.jpg?raw=true) ![MS-Pooling-Module](images/ms_pooling_module.jpg?raw=true)  
  
  
**Setup:**
 * Install Python3.6+
 * Install pip3
 * `pip install -r [requirements.txt](./requirements.txt)`
  
  
**Datasets: (refer to [details](./dataset_details.txt))**
 * IDDAv2 dataset is available on demand;
 * Oxford RobotCar available on the official website. We use the Overcast scenario as the gallery, 
   while the queries are divided into four scenarios: Rain, Snow, Sun, and Night, with one image sampled every 5 meters 
   and filename formatted as @UTMx@UTMy@.jpg . 
  
  
**Usage:**
 * Train: Using the default parameters the script runs the final architecture configuration with 
   ResNet50 encoder, DeepLab semantic segmentation module, multi-scale pooling layer from 4th and 5th conv blocks and 
   finally the domain adaptation module. 
   It follows the exact training protocol and implementation details described into the main paper and the supplementary 
   material. It trains all layers of the encoder and uses the multi-scale attention computed with the features
   extracted from the 4th conv block.  
   `python3 main.py --exp_name=<name output log folder> --dataset_root=<root path of IDDAv2 train dataset> 
   --dataset_root_val=<root path of IDDAv2 val dataset> --dataset_root_test=<root path of RobotCar dataset> 
   --DA_datasets=<path to the RobotCar folder where all scenarios are merged>`  
   To resume the training specify `--resume=<path of checkpoint .pth>`
 * Evaluate:   
   `python3 eval.py --resume=<path of checkpoint .pth> --dataset_root_val=<root path of IDDAv2 val dataset> 
   --dataset_root_test=<root path of RobotCar dataset>`
  
  
**Pretrained models:**
[ResNet50 + DeepLab](weigths/ours_r50_dl.pth)
[ResNet50 + PSPNet](weigths/ours_r50_psp.pth)
[ResNet101 + DeepLab](weigths/ours_r101_dl.pth)
[ResNet101 + PSPNet](weigths/ours_r101_psp.pth)
*Please note:* main paper shows the average recalls running all the experiments three times with different seed. 
Here we provide only one run per configuration.
  
  
**Citation:**
BibTex: 
@article{Paolicelli_2022_ICIAP,   
author = {Paolicelli, Valerio and Tavera, Antonio and Masone, Carlo and Berton, Gabriele Moreno and Caputo, Barbara},   
title = {Learning Semantics for Visual Place Recognition through Multi-Scale Attention},  
booktitle = {}, month = {March}, year = {2022}, pages = {} }