---
title: pytorch环境
date: 2023-10-07 17:24:35
tags: pytorch
categories: deep learning
---

# 配个环境配了一天 人麻了

pytorch+cuda环境配置

```cmd
conda create -n {Env Name} python==3.10
conda activate {Env Name}
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 torchaudio==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

注意使用清华源要关VPN 

注意对应版本[PyTorch/Python/Cuda/torchvision/torchaudio版本对应和兼容性 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/694038606)

实测下来下载速度非常快5~15MB/s 包大概是2~3GB 不超过十分钟下载好 如果不是这样大概率是网络有问题

验证

```
import torch
print(torch.cuda.is_available())
```

应该是True

## 下不下来的包 居然可以直接复制粘贴文件夹吗？（😀

vscode终端出问题。。一直以为是pytorch环境错了

把终端换成（pytorch1）也就是conda的虚拟环境就可以了 而不能用PS的终端
