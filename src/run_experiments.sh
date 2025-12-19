#!/bin/bash

echo "========================================"
echo "开始执行所有实验..."
echo "========================================"

echo "1. 运行 Grocery_and_Gourmet_Food 数据集实验..."
python main.py --model_name MGDCF --dataset Grocery_and_Gourmet_Food --lr 1e-3 --l2 1e-6 --alpha 0.1 --beta 0.9 --n_layers 3
echo "MGDCF 完成"

python main.py --model_name LightGCN --dataset Grocery_and_Gourmet_Food --lr 1e-3 --l2 1e-8 --n_layers 3
echo "LightGCN 完成"

python main.py --model_name BPRMF --dataset Grocery_and_Gourmet_Food --lr 1e-3 --l2 1e-6
echo "BPRMF 完成"

echo "2. 运行 MIND_Large/MINDTOPK 数据集实验..."
python main.py --model_name MGDCF --dataset "MIND_Large/MINDTOPK" --lr 1e-3 --l2 1e-6 --alpha 0.1 --beta 0.9 --n_layers 2
echo "MGDCF 完成"

python main.py --model_name LightGCN --dataset "MIND_Large/MINDTOPK" --lr 1e-3 --l2 1e-8 --n_layers 3
echo "LightGCN 完成"

python main.py --model_name BPRMF --dataset "MIND_Large/MINDTOPK" --lr 1e-3 --l2 1e-6
echo "BPRMF 完成"

echo "========================================"
echo "所有实验执行完毕！"
echo "========================================"