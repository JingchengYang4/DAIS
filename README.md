# DAIS
Depth-Aware Amodal Instance Segmentation Network or DAISnet, utilizes depth information to predict amodal instance segmentation using occlusion relations and 3D shape prior. Depth is an inherent part to occlusion, it provides many useful information for amodal instance segmentation. Occluded objects are always deeper in depth than the occluder, and thus we can deduct which regions are possibly occluded and vice versa. DAISnet utilizes depth information extensively. Depth information can also be used to reconstruct occluded regions. 2D shapes with depth, arguably 2.5D or even 3D, offers feature rich information that could give insight to the position, orientation and region of the object in question. We used a codebook mechanism that uses said features to refine our amodal segmentation. 

## Installation

This project runs on Python 3.6

Install Pytorch 1.4.0 and CUDA 10.1
```
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
```

Install requirements
```
pip install -r requirements.txt
```

Install COCO API
```
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

Setup Detectron2

This is a heavily modified version of Detectron2, newer versions are not yet compatible.

Setup may require you to downgrade to GCC7 temporarily.
```
sudo apt-get install gcc-7 g++-7

sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-7 60 --slave /usr/bin/g++ g++ /usr/bin/g++-7

sudo update-alternatives --config gcc
```

Build Detectron2
```
python -m pip install -e .
```

## Data
```
wget 'GET DATASET FROM KITTI'
mv data_object_image_2.zip DAIS
cd DAIS
mkdir datasets/KINS
mv data_object_image_2.zip datasets/KINS
cd datasets/KINS
unzip data_object_image_2.zip
wget https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset/raw/master/instances_train.json
wget https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset/raw/master/instances_val.json
```


## Training

To train, run the following command
```
python tools/train_net.py --config-file configs/KINS-AmodalSegmentation/mask_rcnn_R_50_FPN_1x_parallel_CtRef_VAR_SPRef_SPRet_FM.yaml
```

To test BTS depth prediction
```
python bts_test.py arguments_test_eigen.txt
```
