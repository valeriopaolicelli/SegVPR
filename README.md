# SegVPR
This is the official PyTorch implementation of our work: "Learning Semantics for Visual Place Recognition through Multi-Scale Attention" accepted at ICIAP 2021.

In this paper we address the task of visual place recognition (VPR), where the goal is to retrieve the correct GPS coordinates of a given query image against a huge geotagged gallery. While recent works have shown that building descriptors incorporating semantic and appearance information is beneficial, current state-of-the-art methods opt for a top down definition of the significant semantic content. Here we present the first VPR algorithm that learns robust global embeddings from both visual appearance and semantic content of the data, with the segmentation process being dynamically guided by the recognition of places through a multi-scale attention module. Experiments on various scenarios validate this new approach and demonstrate its performance against state-of-the-art methods. Finally, we propose the first synthetic-world dataset suited for both place recognition and segmentation tasks.

<!--- [[Paper]](https://openaccess.thecvf.com/content/WACV2022/papers/Tavera_Pixel-by-Pixel_Cross-Domain_Alignment_for_Few-Shot_Semantic_Segmentation_WACV_2022_paper.pdf)
\[[Supplementary]](https://openaccess.thecvf.com/content/WACV2022/supplemental/Tavera_Pixel-by-Pixel_Cross-Domain_Alignment_WACV_2022_supplemental.pdf) --->

Read the paper here: [[ArXiV]](https://arxiv.org/abs/2201.09701)


<table>
<thead>
  <tr>
    <th>Overview</th>
    <th colspan="2">Architecture</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="3"><img src="images/teaser.png" alt="Teaser"/></td>
    <td colspan="2"><img src="images/architecture.jpg" alt="Architecture" width="1154"/></td>
  </tr>
  <tr>
    <td rowspan="2"><img src="images/ms_attention_module.jpg" alt="MS-Attention-Module" width="288"/></td>
    <td rowspan="2"><img src="images/ms_pooling_module.jpg" alt="MS-Pooling-Module" width="288"/></td>
  </tr>
  <tr>
  </tr>
  </tbody>
</table>
  
**Setup:**
 * Install Python3.6+
 * Install pip3
 * `pip install -r `[requirements.txt](./requirements.txt)
  
  
**Datasets: (please refer to [details](./dataset_details.txt))**
 * IDDAv2 dataset NOW available at [Official IDDA Web Site](https://idda-dataset.github.io/home/) ;
 <table>
  <thead>
    <tr>
      <th colspan="2">Town 3</th>
      <th colspan="2">Town 10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="images/g_image_277349.751_110471.756_Town3_.png" alt="Town3_image"/></td>
      <td><img src="images/g_mask_277349.751_110471.756_Town3_.png" alt="Town3_mask"/></td>
      <td><img src="images/g_image_277414.576_110665.787_Town10_.png" alt="Town10_image"/></td>
      <td><img src="images/g_mask_277414.576_110665.787_Town10_.png" alt="Town10_mask"/></td>
    </tr>
    <tr>
      <td colspan="2" style="text-align: center; vertical-align: middle;">UTMx 277349.751<br>UTMy 110471.756<br></td>
      <td colspan="2" style="text-align: center; vertical-align: middle;">UTMx 277414.576<br>UTMy 110665.787<br></td>
    </tr>
  </tbody>
  </table>
 * Oxford RobotCar available on the official website. We use the Overcast scenario as the gallery, 
   while the queries are divided into four scenarios: Rain, Snow, Sun, and Night, with one image sampled every 5 meters 
   and filename formatted as `@UTMx@UTMy@.jpg`. 
  
  
**Usage:**
 * *Train:* Using default parameters, the script uses the final architecture configuration with 
   ResNet50 encoder, DeepLab semantic segmentation module, multi-scale pooling layer from 4th and 5th conv blocks and 
   finally the domain adaptation module. 
   It follows the exact training protocol and implementation details described into the main paper and supplementary 
   material. It trains all layers of the encoder and uses the multi-scale attention computed with the features
   extracted from the 4th conv block.  
   `python3 main.py --exp_name=<name output log folder> --dataset_root=<root path of IDDAv2 train dataset> 
   --dataset_root_val=<root path of IDDAv2 val dataset> --dataset_root_test=<root path of RobotCar dataset> 
   --DA_datasets=<path to the RobotCar folder where all scenarios are merged>`  
   To resume the training specify `--resume=<path of checkpoint .pth>`  
   To change the encoder specify `--arch=resnet101`  
   To change the semantic segmentation module specify `--semnet=pspnet` 
 * *Evaluate:*   
   `python3 eval.py --resume=<path of checkpoint .pth> --dataset_root_val=<root path of IDDAv2 val dataset> 
   --dataset_root_test=<root path of RobotCar dataset>`
  
  
**Pretrained models:**
 * [ResNet50 + DeepLab](https://drive.google.com/file/d/1Jv0hoarx3tTnL59Phl119FoljkJpu9dP/view?usp=sharing)
 * [ResNet50 + PSPNet](https://drive.google.com/file/d/1g33N0gVNGAKWgx0gHbDcAybVxMUkUtdt/view?usp=sharing)
 * [ResNet101 + DeepLab](https://drive.google.com/file/d/1R6m4FpOrf4oOwVO-TnSGZjtE5tR5LCJd/view?usp=sharing)
 * [ResNet101 + PSPNet](https://drive.google.com/file/d/1uQqr7oDeg5T8JQNxtHtEnkleA2orf2cx/view?usp=sharing)  
*Please note:* the main paper shows average recalls obtained from all configurations run three times with different seeds respectively. 
Here instead we provide only one model per configuration.
  
## Cite us
If you use this repository, please consider to cite us:

    @InProceedings{Paolicelli_2022_ICIAP,
    author    = {Paolicelli, Valerio and Tavera, Antonio and Masone, Carlo and Berton, Gabriele Moreno and Caputo, Barbara}},
    title     = {Learning Semantics for Visual Place Recognition through Multi-Scale Attention},
    booktitle = {Image Analysis and Processing ??? ICIAP 2022},
    month     = {},
    year      = {},
    pages     = {}
