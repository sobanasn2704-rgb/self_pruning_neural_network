#  Self-Pruning Neural Network

##  Overview

This project implements a neural network that learns to prune its own weights during training using learnable gates.
Trained on CIFAR-10, it demonstrates the trade-off between accuracy and sparsity.



##  Key Ideas

* **Straight-Through Estimator (STE):** Enables binary pruning (0/1 gates) with gradient flow
* **Adaptive λ Scheduling:** Delays pruning to preserve learning
* **Layer-wise Sparsity:** Later layers pruned more aggressively

##  Model

MLP with custom **PrunableLinear layers** + BatchNorm + Dropout.


##  Loss

Loss = CrossEntropy + λ × Sparsity Loss (mean of gate values)

##  Observation

Model learns a **bimodal gate distribution** → many weights pruned (≈0) while important ones remain active.

##  Run


pip install torch torchvision matplotlib
python train.py




The network **learns its optimal structure during training**, not after — making it efficient and adaptive.
