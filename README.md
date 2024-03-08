
# Generalizable Two-Branch Framework for Image Class Incremental Learning （ICASSP'24）

This repository is Pytorch code for our proposed Generalizable Two-Branch Framework for Image Class Incremental Learning (G2B). 

Paper link: https://arxiv.org/abs/2402.18086



## 1.Environment Setup
The code and models were tested on Linux Platform with two GPU (RTX3080Ti).
First creating a conda environment with all the required packages by using the following command.

```
conda env create -f environment.yml
```
It creates a conda environment named `G2B`. You can activate the conda environment with the command:
```
conda activate G2B
```
In the following sections, we assume that this conda environment is in use.

*Potential Compatibility Issues:*
1.If you see the following error, it usually mean the  PyTorch package incompatible with the infrastructure.
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
For example, your machine supports CUDA == 11.1, install a PyTorch package using CUDA11.1 instead:
```
pip uninstall torch
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```


## 2.Run
Training and testing of the proposed method are reproduced on CIFAR100 10-task class incremental learning (CIL):
### G2B(DER)
```
# Model Training
cd ./CNN
python main.py --config=./exps/cifar100/G2B_DER.json
```
### G2B(DyTox)
```
# Model Training
cd ./VIT
bash train.sh 0,1 \
  --options options/data/cifar100_10-10.yaml options/data/cifar100_order1.yaml options/model/cifar_dytox.yaml \
  --name G2B_dytox \
  --data-path ./data \
  --output-basedir ./checkpoints \
  --memory-size 2000 \
  --add_mask
 ```
## 3.Results
The reproduced results of CIFAR100 10-task CIL:

Task | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | Avg. 
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: 
G2B(DER)| **94.6** | **87.75** | **82.23** | **77.88** | **76.8** | **75.55** | **74.68** | **73.24** | **71.03** | **68.98** |**78.26**
G2B(DyTox)| **90.9** | **88.25** | **83.67** | **79.22** | **77.74** | **71.3** | **69.17** | **65.45** | **63.49** | **62.04** | **75.12**
Note: Different pytorch versions may lead to slightly different results. (pytorch ver. >= 1.8.1 required)

## Citation
If you find this code useful, please kindly cite the following paper:
<pre>
@article{wu2024general,
  title={Generalizable Two-Branch Framework for Image Class Incremental Learning},
  author={Wu, Chao and Chang, Xiaobin and Wang, Ruixuan},
  booktitle = {IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)},
  year={2024}
}
</pre>
