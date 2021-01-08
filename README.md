# SSA2D
Code for Single Shot Actor-Action Detection in Videos (SSA2D). 
Paper accessible at [WACV 2021 Proceedings] (https://openaccess.thecvf.com/content/WACV2021/papers/Rana_We_Dont_Need_Thousand_Proposals_Single_Shot_Actor-Action_Detection_in_WACV_2021_paper.pdf)

## Description
This is an implementation of SSA2D on A2D and VidOR datasets. It is built using the Keras library. The datasets have to be downloaded separately.

### Setup
Download the dataset and assign their paths in respective dataloader. For VidOR, assign data and annotation path in dataloader_vidor.py file.
VidOR uses some processed annotation files, available in data folder.

### Training
Run train_ssa2d_vidor.py to train SSA2D on VidOR. The trained models and log files will be saved in specified folders.

### To-Do
1. Add clean code for A2D dataloader.
2. Add clean code for full evaluation.
3. Generalized dataset format for both datasets.