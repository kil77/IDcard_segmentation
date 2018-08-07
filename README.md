# DeepLabV3 Semantic Segmentation
Reimplementation of DeepLabV3 Semantic Segmentation

This is an (re-)implementation of [DeepLabv3 -- Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587) in TensorFlow for semantic image segmentation on the [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/). The implementation is based on [DrSleep's implementation on DeepLabV2](https://github.com/DrSleep/tensorflow-deeplab-resnet) and [CharlesShang's implementation on tfrecord](https://github.com/CharlesShang/FastMaskRCNN).

## Features
- [x] Tensorflow support
- [ ] Multi-GPUs on single machine (synchronous update)
- [ ] Multi-GPUs on multi servers (asynchronous update)
- [x] ImageNet pre-trained weights
- [ ] Pre-training on MS COCO
- [x] Evaluation on VOC 2012
- [ ] Multi-scale evaluation on VOC 2012

## Requirement
#### Tensorflow 1.4
```
python 3.5
tensorflow 1.4
CUDA  8.0
cuDNN 6.0
```

#### Tensorflow 1.2
```
python 3.5
tensorflow 1.2
CUDA  8.0
cuDNN 5.1
```
The code written in Tensorflow 1.4 are compatible with Tensorflow 1.2, tested on single GPU machine.

#### Installation
```
sh setup.sh
```

## Train
1. Configurate `config.py`.
2. Run `python3 convert_voc12.py --split-name=SPLIT_NAME`, this will generate a tfrecord file in `$DATA_DIRECTORY/records`.
3. Single GPU: Run `python3 train_voc12.py` (with validation mIOU every SAVE_PRED_EVERY).


## Performance
This repository only implements MG(1, 2, 4), ASPP and Image Pooling. The training is started from scratch. (The training took me almost 2 days on a single GTX 1080 Ti. I changed the learning rate policy in the paper: instead of the 'poly' learning rate policy, I started the learning rate from 0.01, then set fixed learning rate to 0.005 and 0.001 when the seg_loss stopped to decrease, and used 0.001 for the rest of training. )

### Updated 1/11/2018
I continued training with learning rate 0.0001, there is a huge increase on validation mIOU.

### Updated 2/05/2018
There was an improvement on the implementation of Multi-grid, thanks @howard-mahe. The new validation results should be updated soon.

### Updated 2/11/2018
The new validation result was trained from scratch. I didn't implement the two stage training policy (fixing BN and stride 16 -> 8). I may try few more runs to see if there is an improvement on the performance, but I think it is a fine-tuning work.

| mIOU      | Validation       |
| --------- |:----------------:|
| paper     | 77.21%           | 
| repo      | 70.63%           |

The validation mIOU for this repo is achieved without multi-scale and left-right flippling.

The improvement can be achieved by finetuning on hyperparameters such as **learning rate**, **batch size**, **optimizer**, **initializer** and **batch normalization**. I didn't spend too much time on training and the results are temporary. 

*Welcome to try and report your numbers.*
