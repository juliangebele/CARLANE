# UFLD-ADDA
Official PyTorch implementation of UFLD-ADDA used in the paper "CARLANE: A Lane Detection Benchmark for Unsupervised Domain Adaptation from Simulation to multiple Real-World Domains".


## Environment
- Python 3.8 
- PyTorch 1.8.1

## Installation
Install the required python modules with

```Shell
pip install -r requirements.txt
```

## Getting started
First, modify necessary parameters in your `configs/molane.py`, `configs/tulane.py` or `configs/mulane.py` config according to your environment. 
- `data_root` main path of your dataset, where your train, validation and test images are stored. 
- `log_path` this is where trained models and code backup are stored. ***It should be placed outside of this project.***
- `pretrained` path of the pretrained UFLD-SO model.
- `source_train, etc.` path of the .txt files for training, validation and test data


For single gpu training, run
```Shell
python train.py configs/path_to_your_config
```
For multi-gpu training, run
```Shell
python -m torch.distributed.launch --nproc_per_node=$NGPUS train.py configs/path_to_your_config
```
If there is no pretrained torchvision model, multi-gpu training may result in multiple downloading. You can first download the corresponding models manually, and then restart the multi-gpu training.

## Evaluation
Use the evaluation code inside [UFLD-SGPCS](https://github.com/juliangebele/CARLANE/blob/master/CARLANE%20Baselines/UFLD-SGPCS/pcs/test.py) to quantitatively and qualitatively evaluate the model. 

## Acknowledgement
This code is based on [Ultra Fast Structure-aware Deep Lane Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection) and [ADDA.PyTorch](https://github.com/Fujiki-Nakamura/ADDA.PyTorch). We thank the authors for sharing their code.
