## Introduction
本项目旨在基于ReChorus框架实现MGDCF模型的复现。
MGDCF模型是一种基于图神经网络的推荐系统模型，其主要思想是将用户的和项目之间的关系建模为图，并通过图神经网络学习用户的兴趣图表示，从而预测用户对项目的喜好程度。
MGDCF通过应用马尔科夫图神经网络（MGDN）和 InfoBPR 损失函数，可以有效地捕捉用户和项目之间的复杂关系，并学习到用户的兴趣图表示，提升了原有模型的性能。

## Code
本项目基于ReChorus框架实现了MGDCF模型的复现。
`MGDCF.py`文件位于`ReChorus/src/models/general`目录下，实现了MGDCF模型的初始化、数据编码、前向传播、损失函数计算等功能。
`MGDCFBase`类实现了MGDCF模型的基本结构，包括邻接矩阵、关系矩阵等，并通过`MGDCFEncoder`类实现对数据的编码处理。
`MGDCF`类继承自`MGDCFBase`类，其中的`loss`函数通过InfoBPR损失和l2损失来实现模型损失的计算。 
`MGDCFEncoder`类实现了对用户和项目的特征编码，包括用户特征编码、项目特征编码等。通过运用马尔科夫图神经网络（MGDN）提升了模型的性能。

可以直接下载本项目的`ReChorus`文件夹，并运行我们下面给出的代码，即可复现MGDCF模型。也可以只下载`MGDCF.py`文件，将其放在下载好的ReChorus框架中（具体位置为`ReChorus/src/models/general`），然后运行我们下面给出的代码。

## Requirements
```
python == 3.10.4
torch == 2.5.0
pandas == 1.4.4
scipy == 1.15.3
tqdm == 4.67.1
scikit-learn
pyyaml
numpy

```
## training
使用 GPU 训练：将 --gpu -1 改为 --gpu 0（使用第一个GPU）
使用 CPU 训练：保持 --gpu -1（默认）
所有训练均在 Linux 环境下测试通过，训练时间较长，请耐心等待。
如果硬件条件受限，可以仅在cpu环境下进行训练。如果有gpu环境，可以将 `--gpu`后面的参数设为0，使用gpu进行训练。
训练的数据位于 `ReChorus/data`，我们提供两个数据集进行TOP-K训练任务，分别是`Grocery_and_Gourmet_Food`和`MIND_Large/MINDTOPK`。输出放在`ReChorus/log`目录下。
下面是我们对不同模型的训练命令：
```cd src```<br>
对于`Grocery_and_Gourmet_Food`数据集：<br>
MGDCF:```python main.py --gpu -1 --model_name MGDCF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food --alpha 0.1 --beta 0.9 --n_layers 3```<br>
LightGCN:```python main.py --gpu -1 --model_name LightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8 --dataset Grocery_and_Gourmet_Food```<br>
BPRMF:```python main.py --gpu -1 --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset Grocery_and_Gourmet_Food```<br>
对于`MIND_Large/MINDTOPK`数据集：<br>
MGDCF:```python main.py --gpu -1 --model_name MGDCF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MIND_Large/MINDTOPK --alpha 0.1 --beta 0.9 --n_layers 2```<br>
LightGCN:```python main.py --gpu -1 --model_name LightGCN --emb_size 64 --n_layers 3 --lr 1e-3 --l2 1e-8 --dataset MIND_Large/MINDTOPK```<br>
BPRMF:```python main.py --gpu -1 --model_name BPRMF --emb_size 64 --lr 1e-3 --l2 1e-6 --dataset MIND_Large/MINDTOPK```<br>
所有训练均在Linux环境下进行，也可以在Windows下运行，训练时间较长，请耐心等待。
我们提供了批量运行脚本 run_experiments.sh，可以一次性运行所有实验:```chmod +x run_experiments.sh```<br>
```./run_experiments.sh```<br>
## Supplementary Experiments (补充实验)
为了深入分析MGDCF模型，我们补充了**超参数实验**与**消融实验**。所有补充实验均在 `Grocery_and_Gourmet_Food` 数据集上进行。
### 1. 超参数实验和消融实验
我们已经运行了 `n_layers=3` 作为主实验。以下是补充运行的命令，用于对比分析：<br>
Layer = 1:```python main.py --model_name MGDCF --dataset Grocery_and_Gourmet_Food --lr 1e-3 --l2 1e-6 --alpha 0.1 --beta 0.9 --n_layers 1 --epoch 50```<br>
Layer = 2:```python main.py --model_name MGDCF --dataset Grocery_and_Gourmet_Food --lr 1e-3 --l2 1e-6 --alpha 0.1 --beta 0.9 --n_layers 2 --epoch 50```<br>
Layer = 4:```python main.py --model_name MGDCF --dataset Grocery_and_Gourmet_Food --lr 1e-3 --l2 1e-6 --alpha 0.1 --beta 0.9 --n_layers 4 --epoch 50```<br>
运行消融实验 (No Diffusion: alpha=1.0, beta=0.0):```python main.py --model_name MGDCF --dataset Grocery_and_Gourmet_Food --lr 1e-3 --l2 1e-6 --alpha 1.0 --beta 0.0 --n_layers 3 --epoch 50```<br>
## parameter
可能要用到的参数：
gpu $\Rightarrow$ gpu的编号: 使用该编号的gpu进行训练
model_name $\Rightarrow$ 模型名称: 使用该模型进行训练
dataset $\Rightarrow$ 数据集名称: 使用该数据集进行训练
lr $\Rightarrow$ 学习率: 学习率
l2 $\Rightarrow$ l2正则化系数: l2正则化系数
n_layers $\Rightarrow$ 图神经网络层数: 图神经网络层数
alpha $\Rightarrow$ $\alpha$系数: 仅在MGDCF模型中使用
beta $\Rightarrow$ $\beta$系数: 仅在MGDCF模型中使用

---------------------------------------------------------
ReChorus框架代码地址：```https://github.com/THUwangcy/ReChorus```
如果在运行代码过程中遇到什么问题，请在企业微信上联系我们。
本项目基于 ReChorus 框架开发，感谢原作者的优秀工作
