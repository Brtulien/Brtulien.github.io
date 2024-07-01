---
title: knowledge-distillation
date: 2024-06-27 10:54:11
tags: [Attention, Transformer, deep learning]
categories: deep learning
---

~~偷懒失败，还是写一下~~

# 概念

知识蒸馏就是把一个大的模型当作教师模型，然后把他的知识教给较小的模型（学生模型）。

<img src="../images/$%7Bfiilename%7D/image-20240627105847582.png" alt="image-20240627105847582" style="zoom:50%;" />

大的模型较臃肿，真正落地的中断算力有限，比如手表等，通过知识蒸馏把大模型变为小模型，再把小模型部署到终端。

# 模型

## soft target

学生网络有两种标签，一种是**教师网络的输出**（Soft-target），一种是**真实的标签**（Hart-target）。

soft target就是常用的概率标签，比如

![image-20240627111036662](../images/$%7Bfiilename%7D/image-20240627111036662.png)

hard target 预测为马1 驴0 汽车0

soft target 预测为马0.7 驴0.25 汽车0.05

传统的神经网络的训练方法是定义一个损失函数，目标是使预测值尽可能接近于真实值，使用原始数据集标注的one-shot标签（也就是Hard-Target）。损失函数就是使神经网络的损失值的和尽可能小，这种训练过程是对ground truth(真实值)求极大似然.。

但是知识蒸馏中使用的是Teacher模型softmax层输出的类别概率作为Soft-Target。因为Soft-target中包含了Teacher模型训练和推理的大量信息，比如某些负标签概率远大于其他负标签，说明该样本与该负标签之间有一定的相似性。因此知识蒸馏中每个样本带给Student模型的信息大于传统的训练方式。

如在MNIST数据集中做手写体数字识别任务，假设某个输入的“2”更加形似"3"，softmax的输出值中"3"对应的概率会比其他负标签类别高；而另一个"2"更加形似"7"，则这个样本分配给"7"对应的概率会比其他负标签类别高。这两个"2"对应的Hard-target的值是相同的，但是它们的Soft-target却是不同的，由此我们可见Soft-target蕴含着比Hard-target更多的信息。

<img src="../images/$%7Bfiilename%7D/640.webp" alt="图片" style="zoom:50%;" />

在使用 Soft-target 训练时，Student模型可以很快学习到 Teacher模型的推理过程；而传统的 Hard-target 的训练方式，所有的负标签都会被平等对待。因此，Soft-target 给 Student模型带来的信息量要大于 Hard-target，并且Soft-target分布的熵相对高时，其Soft-target蕴含的知识就更丰富。同时，使用 Soft-target 训练时，梯度的方差会更小，训练时可以使用更大的学习率，所需要的样本也更少。这也解释了为什么通过蒸馏的方法训练出的Student模型相比使用完全相同的模型结构和训练数据只使用Hard-target的训练方法得到的模型，拥有更好的泛化能力。

## 蒸馏温度

### Logits

比如图片分类，Logits就是最后输出的（softmax之前的）信息。softmax之后得到概率分布。

通过蒸馏来调整输出的概率分布，因为当softmax输出的概率分布较小的时候，负标签的值都很接近0，对损失函数的贡献小到可以忽略，因此要通过**蒸馏温度**来调整softmax的输出。

​				$$q_i=\frac{exp(z_i/T)}{\sum_{j}exp(z_j/T)}$$

T越高，softmax的输出就越趋于平滑，分布的熵就越大，负标签携带的信息也会被放大，模型训练就会更加关注负标签。

加入T还有哪些作用：

1. **抑制过拟合：** 高蒸馏温度下的软目标概率分布更平滑，相比硬目标更容忍学生模型的小误差。这有助于防止学生模型在训练过程中对教师模型的一些噪声或细微差异过度拟合，提高了模型的泛化能力。
2. **降低标签噪声的影响：** 在训练数据中存在标签噪声或不确定性时，平滑的软目标可以减少这些噪声的影响。学生模型更倾向于关注教师模型输出的分布，而不是过于依赖单一的硬目标。
3. **提高模型鲁棒性：** 平滑的软目标有助于提高模型的鲁棒性，使其对输入数据的小变化更加稳定。这对于在实际应用中面对不同环境和数据分布时的模型性能至关重要。

## 训练过程

知识蒸馏训练的具体方法如下图所示，主要包括以下几个步骤：

1. 训练好Teacher模型；
2. 利用高温$T_{high}$ 产生 Soft-target；
3. 使用$\{Soft-target,T_{high}\}$ 和$\{Hard-target,T_{high}\}$ 同时训练 Student模型；
4. 设置温度T=1，Student模型线上做inference。

<img src="../images/$%7Bfiilename%7D/640-17194607142405.webp" alt="图片" style="zoom:50%;" />



## 损失函数

学生网络既要在蒸馏温度等于T时与教师网络的结果相接近。 也要保证不使用蒸馏温度时的结果与真实结果相接近。

蒸馏损失（distill loss）：把教师网络用蒸馏温度T输出的结果与学生网络蒸馏温度为T的输出结果做损失，这个损失越小越好。$L_{soft}=-\sum^{N}_{i}p_i^Tlog(q_i^T)$

其中$p_i^T=\frac{exp(v_i/T)}{\sum_k^Nexp(v_k/T)}$

$q_i^T=\frac{exp(z_i/T)}{\sum_k^Nexp(z_k/T)}$

其实就是蒸馏温度T下教师和学生的损失。

学生损失（Student loss）：学生网络蒸馏温度为1的预测结果和真实的标签做loss。

$L_{hard=-\sum_i^Nc_ilog(q_i^1)}$

注意理解$L_{soft}和L_{hard}$的关系：log前面是标签  里面是蒸馏温度T的softmax。

最后将二者加权求和$L=\alpha L_{soft}+\beta L_{hard}$

当$L_{hard}$权重较小的时候往往能取得较好的效果。

## 推理

训练好后把X输入到学生网络进行推理。



### 参考

[深度学习中的知识蒸馏技术 (qq.com)](https://mp.weixin.qq.com/s?__biz=MzU0MDQ1NjAzNg==&mid=2247527921&idx=2&sn=1894bde4de9ecdbb2b31fa2a64cc6551&chksm=fb3ac8facc4d41ec0ab8e9d0d865163524ad741ae9dc8b0e57c269e54894abbce0e0bcd37a3d&scene=27)

[全网最细图解知识蒸馏(涉及知识点：知识蒸馏实现代码，知识蒸馏训练过程，推理过程，蒸馏温度，蒸馏损失函数)-CSDN博客](https://blog.csdn.net/qq_42864343/article/details/134693835)

*扩展一下轻量化网络的四个方向*

<img src="../images/$%7Bfiilename%7D/308a7daaa10e42c6a2c6a2778eaddca7.png" alt="在这里插入图片描述" style="zoom:50%;" />
