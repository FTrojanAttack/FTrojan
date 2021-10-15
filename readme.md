# Backdoor Attack through Frequency Domain

## DEPENDENCIES
python==3.8.3  
numpy==1.19.4  
tensorflow==2.4.0  
opencv==4.5.1  
idx2numpy==1.2.3  
pytorch==1.7.0  

## Dataset Preparation
We provide CIFAR10 frequency attack version. GTSRB, ImageNet, and PubFig can be easily modified by this project. 


## Change Config
You can modify the *param dict* in the train.py file, and the th_train.py file to train your own backdoored model. 

There are 6 parameters as follows:   
* dataset: CIFAR10
* target_label: The target label to backdoor. Default: 8  
* poisoning_rate: The rate of poisoning sample. A float number ranging (0,1)  
* channel_list: Which channels to implant backdoor, [1,2] means UV, [0,1,2] means YUV.
* magnitude: The magnitude of the trigger. There are two ways to implant the trigger, 
  first is to add a fix value onto one frequency. Second is to set one frequency to a fix value. 
  The effectiveness of the two ways are same.
  
* YUV: True, YUV Channel, False, RGB Channel
* pos_list: the position of the trigger in the frequency map



## Run Backdoor Attack Code
Tensorflow2.0:
```shell
python train.py
```

Pytorch:
```shell
python th_train.py
```


