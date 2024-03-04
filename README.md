# 3dvg_SeCG
The source code of paper Semantic-Enhanced 3D Visual Grounding via Cross-modal Graph Attention

## Environment
### Requirements
- CUDA: >=11.3  
- Python: >=3.8  
- PyTorch: >=1.12.0  
### Installation
```
pip install h5py
pip install transformers
pip install pickle
pip install tensorboardX

cd external_tools/pointnet2
python setup.py install
```
### Data Preparation
Download the [ScanNet V2](http://www.scan-net.org/) dataset.  
Prepare for ScanNet data and package it into .pkl
```
cd data
python prepare_scannet_data.py
```

### Pretrained Model
