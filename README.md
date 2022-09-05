# DAAIS
Depth Aware Amodal Instance Segmentation Network

---

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

Setup Detectron2
This is a heavily modified version of Detectron2, newer versions are not compatible.
```
python -m pip install -e .
```

Install COCO API
```
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```