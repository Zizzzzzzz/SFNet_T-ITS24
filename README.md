# Enhancing Traffic Object Detection in Variable Illumination with RGB-Event Fusion

<p align="center">
  <img src="https://github.com/Zizzzzzzz/SFNet_2024/blob/main/imgs/intro.jpg" width="750">
</p>

This is the official Pytorch implementation of the IEEE T-ITS 2024 paper [Enhancing Traffic Object Detection in Variable Illumination with RGB-Event Fusion](https://ieeexplore.ieee.org/document/10682110).

## Abstract
Traffic object detection under variable illumination is challenging due to the information loss caused by the limited dynamic range of conventional frame-based cameras. To address this issue, we introduce bio-inspired event cameras and propose a novel Structure-aware Fusion Network (SFNet) that extracts sharp and complete object structures from the event stream to compensate for the lost information in images through cross-modality fusion, enabling the network to obtain illumination robust representations for traffic object detection. Specifically, to mitigate the sparsity or blurriness issues arising from diverse motion states of traffic objects in fixed-interval event sampling methods, we propose the Reliable Structure Generation Network (RSGNet) to generate Speed Invariant Frames (SIF), ensuring the integrity and sharpness of object structures. Next, we design a novel Adaptive Feature Complement Module (AFCM) which guides the adaptive fusion of two modality features to compensate for the information loss in the images by perceiving the global lightness distribution of the images, thereby generating illumination-robust representations. Finally, considering the lack of large-scale and high-quality annotations in the existing event-based object detection datasets, we build a DSEC-Det dataset, which consists of 53 sequences with 63,931 images and more than 208,000 labels for 8 classes. Extensive experimental results demonstrate that our proposed SFNet can overcome the perceptual boundaries of conventional cameras and outperform the frame-based methods, e.g., YOLOX by 7.9% in mAP50 and 3.8% in mAP50:95.

## Installation (same as [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX))

```Bash
pip3 install -v -e .  # or  python3 setup.py develop
```

## DSEC-Det dataset
<p align="center">
  <img src="https://github.com/Zizzzzzzz/SFNet_2024/blob/main/imgs/dataset.jpg" width="750">
</p>

To evaluate or train SFNet you will need to download the required preprocessed datasets:

Link: https://pan.baidu.com/s/15M7pUKyMepAUumwLSR6-MQ  Password: ryx6

Labels: [datasets](https://github.com/Zizzzzzzz/SFNet_2024/tree/main/datasets).

The official DSEC is available here: [ https://dsec.ifi.uzh.ch](https://dsec.ifi.uzh.ch).

Details can be found in the paper [ DSEC: A Stereo Event Camera Dataset for Driving Scenarios](https://rpg.ifi.uzh.ch/docs/RAL21_DSEC.pdf).

The code for homographic transformation of RGB images to Event frame can be found here:
[homographic tansformation](https://github.com/RunqiuBao/fov_alignment/blob/main/fov_align.ipynb)

## Evaluation

- Run ./datasets/convert_dsecdet_to_yolox.py to generate COCO json files.

- Download RSGNet model.
Link: https://drive.google.com/file/d/1P3lSLccasQ79EKmMk-MpAb4Wo9rU-XNq/view?usp=drive_link

Generating SIF:
```Bash
cd RSGNet
python3 test.py
```

- Download model.
Link: https://pan.baidu.com/s/1yPdmGKEs_tpeeI7gI0Subw
Password: jtd7

all8
```Bash
python3 -m yolox.tools.eval -f exps/example/DSEC-Det/test_all.py -c ckpt/all8_best.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
```
all2
```Bash
python3 -m yolox.tools.eval -f exps/example/DSEC-Det/test_all_2classes.py -c ckpt/all2_best.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
```
sub8
```Bash
python3 -m yolox.tools.eval -f exps/example/DSEC-Det/test_sub.py -c ckpt/sub8_best.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
```
sub2
```Bash
python3 -m yolox.tools.eval -f exps/example/DSEC-Det/test_sub_2classes.py -c ckpt/sub2_best.pth -b 1 -d 1 --conf 0.001 --fp16 --fuse
```

```bibtex
@article{liu2024enhancing,
  title={Enhancing traffic object detection in variable illumination with rgb-event fusion},
  author={Liu, Zhanwen and Yang, Nan and Wang, Yang and Li, Yuke and Zhao, Xiangmo and Wang, Fei-Yue},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  year={2024},
  publisher={IEEE}
}
```

## Code Acknowledgments
This project has used code from the following projects:
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
