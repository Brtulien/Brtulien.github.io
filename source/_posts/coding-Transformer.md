---
title: coding-Transformer
date: 2024-06-25 14:19:15
tags: [Attention, Transformer, deep learning, coding]
categories: deep learning
---

~~不想打公式 直接截图了（苦鲁西~~

# 代码

目录：/fairseq/models/transformer/

### transformer_legacy.py

226行注册了transformer_model_architecture，可以自定义配置。

同文件22行注册了transformer。继承TransformerModelBase。

77行init，先从Config中读取配置，再从基类初始化，最后初始化参数args

```py
def __init__(self, args, encoder, decoder):
        cfg = TransformerConfig.from_namespace(args)
        super().__init__(cfg, encoder, decoder)
        self.args = args
```

92行build_model主要是根据参数配置：

encoder decoder层的数量；

source和target的最大编码长度；

检查和设置共享嵌入

之后的方法都直接调用基类。



### Transformer_config.py

一些参数配置

### Transformer_base.py

forward中为整体的流程：调用encoder和decoder最后输出decoder_out

### transformer_encoder.py

init中需要初始化一系列参数 比如embed_positions，layernorm_embedding等

build_encoder_layer建立encoder层发现要跳转道Transformer Encoder Layer Base

forward_embedding进行一系列嵌入（包括位置嵌入）

max_position为设置的最大输入长度

## Encoder

在/fairseq/modules/transformer_layer.py

### forward

首先设置注意力掩码

```py
if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool), -1e8 if x.dtype == torch.float32 else -1e4
            )
# 保存残差
residual = x
```

然后是自注意力层

```py
if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
```

dropout正则化 防止过拟合

```py
x = self.dropout_module(x)
```

Add & Norm层

Add层参考残差网络 防止退化

Norm层归一化

```py
# 残差连接 将原来的残差与新的x相加
x = self.residual_connection(x, residual)
if not self.normalize_before:
    x = self.self_attn_layer_norm(x)
# 保存残差
residual = x
# 归一化
if self.normalize_before:
    x = self.final_layer_norm(x)
```

激活函数和全连接层

再做一次Add & Norm层

并返回结果

```py
x = self.activation_fn(self.fc1(x))
x = self.activation_dropout_module(x)
# 全连接
x = self.fc2(x)
# 保存全连接层输出
fc_result = x
# 再做一次Add & Norm
x = self.dropout_module(x)
x = self.residual_connection(x, residual)
if not self.normalize_before:
    x = self.final_layer_norm(x)

        if self.return_fc and not torch.jit.is_scripting():
            return x, fc_result
        return x
```

用到的一些函数如下：

### 全连接层

```py
def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
    return quant_noise(
        nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
    )
```

### 自注意力层

```py
def build_self_attention(self, embed_dim, cfg):
    return MultiheadAttention(
        embed_dim,
        cfg.encoder.attention_heads,
        dropout=cfg.attention_dropout,
        self_attention=True,
        q_noise=self.quant_noise,
        qn_block_size=self.quant_noise_block_size,
        xformers_att_config=cfg.encoder.xformers_att_config,
    )
```

### 残差连接

```py
def residual_connection(self, x, residual):
    return residual + x
```

## Decoder

### forward

先设置自注意力的状态和输入缓冲

```py
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
```

设置掩码

```py
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

```

自注意力

```py
        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
```

Add & Norm

```py
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)
```

Encoder_attn层 也是多头自注意力但是输入是前一层的输出和Encoder的输出结合

```py
        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
```

最后Add&Norm激活全连接归一化返回结果



## Multi-Head Attention

在module的Multi-Head Attention中 ~~太长了看不懂 先鸽了~~

# 题目

## BPE

BPE（Byte Pair Encoding）是字节对编码，在固定大小的 词表中实现可变长度的子词。将词分成单个字符，然后依次用另一个字符替换频率最高的一对字符，直到循环次数结束。

算法流程：

1. 准备语料库  确定期望的subword词表大小等参数

2. 在每个单词末尾添加后缀</w> 统计词频 如“l o w</w>”:5

3. 将所有**单词拆分成单个字符** 用所有单个字符建立最初的词典 并统计单个字符的频率

4. 挑出频次最高的符号对 比如t h组成th 将新字符加入词表 然后merge 将所有的t h变为th（有点类似哈夫曼树）

重复 上述操作 直到词表中单词数达到设定量或下一个最高频数为1 达到设定量后其余词汇直接丢弃

BPE可以有效地平衡词典大小和编码步骤数。

参考[BPE 算法原理及使用指南【深入浅出】-CSDN博客](https://blog.csdn.net/a1097304791/article/details/122068153)



## Embedding

```py
class PositionalEncoding(nn.Module):
def __init__(self, d_model, dropout, max_len=5000):
"""
:param d_model: pe编码维度，一般与word embedding相同，方便相加
:param dropout: dorp out
:param max_len: 语料库中最长句子的长度，即word embedding中的L
"""
super(PositionalEncoding, self).__init__()
# 定义drop out
self.dropout = nn.Dropout(p=dropout)
# 计算pe编码
pe = torch.zeros(max_len, d_model) # 建立空表，每行代表一个词的位置，每列代
表一个编码位
position = torch.arange(0, max_len).unsqueeze(1) # 建个arrange表示词的
位置以便公式计算，size=(max_len,1)
div_term = torch.exp(torch.arange(0, d_model, 2) * # 计算公式中
10000**（2i/d_model)
-(math.log(10000.0) / d_model))
pe[:, 0::2] = torch.sin(position * div_term) # 计算偶数维度的pe值
pe[:, 1::2] = torch.cos(position * div_term) # 计算奇数维度的pe值
pe = pe.unsqueeze(0) # size=(1, L, d_model)，为了后续与word_embedding
相加,意为batch维度下的操作相同
self.register_buffer('pe', pe) # pe值是不参加训练的
def forward(self, x):
# 输入的最终编码 = word_embedding + positional_embedding
x = x + Variable(self.pe[:, :x.size(1)],requires_grad=False) #size =
[batch, L, d_model]
return self.dropout(x) # size = [batch, L, d_model]
```

## Cross-attention和Self-attention的区别

Cross-attention也就是代码中的encoder-decoder-attention

```py
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beam_size > 1 and bsz == key.size(1):
                    # key is [T, bsz*beam_size, C], reduce to [T, bsz, C]
                    key = key.view(key.size(0), -1, self.beam_size, key.size(2))[
                        :, :, 0, :
                    ]
                    if key_padding_mask is not None:
                        key_padding_mask = key_padding_mask.view(
                            -1, self.beam_size, key_padding_mask.size(1)
                        )[:, 0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)

```

交叉注意力用于decoder中使其当前状态和encoder的output交互。decoder作为Q encoder_output为K、V

## BLEU指标

### N-gram

BLEU指标采用N-gram的匹配机制，就是比较译文和参考译文之间n组词之间相似的一个占比。

<img src="../images/$%7Bfiilename%7D/626346-20171016222256849-1802531988.png" alt="img" style="zoom:;" />

译文分为4个3-gram词组 有2个命中参考译文 则该译文的3-gram匹配度为2/4

### 召回率

机器译文：the the the the 

人工译文：The cat is standing on the ground

1-gram的匹配度为1但是the在参考译文只出现了2次，如果匹配度直接用1很显然是不合理的。

$Count_{clip}=min(Count,Max\_Ref\_Count)$前者为译文中出现的次数，后者为参考译文中的最大次数，取最小值限制上文的情况（即最多不超过参考译文中该单词的频率）



计算每个N-gram的公式如下：

人工译文$s_j$

机器译文$c_i$

$h_k(c_i)$表示第k个词组在翻译译文$c_i$出现的次数

$h_k(s_{i,j})$表示第k个词组在标准答案$s_{i,j}$出现的次数$

![image-20240626192902992](../images/$%7Bfiilename%7D/image-20240626192902992.png)

### 惩罚因子

N-gram的匹配度可能会随着句子长度变短而变好，因此会存在一个问题，一个翻译引擎只翻译出部分句子且比较准确，那么匹配度依然很高。因此要引入长度惩罚因子

<img src="../images/$%7Bfiilename%7D/image-20240626194522680.png" alt="image-20240626194522680" style="zoom: 50%;" />

$l_c$代表机器译文的长度

$l_s$代表参考译文的有效长度

当机器译文较长的时候不惩罚

最终公式

<img src="../images/$%7Bfiilename%7D/image-20240626194923426.png" alt="image-20240626194923426" style="zoom:50%;" />

BLEU采用均匀加权 $W_n=1/N$

N最大为4 即最多4-gram

参考ca[BLEU算法（例子和公式解释）-CSDN博客](https://blog.csdn.net/qq_30232405/article/details/104219396)

## beam

Beam Search

Greedy Search问题在于在每一步它只选择得分最高的top 1单词，假设被它忽略的top 2单词带来的后面一系列单词使得整个序列的得分反而更高，则Greedy Search就不会得到最合理的解码结果。
Beam Search集束搜索是Greedy Search的改进版，它拓展了Greedy Search在每一步的搜索空间，每一步保留当前最优的K个候选，一定程度上缓解了Greedy Search的问题，令K为Beam Size代表束宽，Beam Size是一个超参数，它决定搜索空间的大小，**越大搜索结果越接近最优，但是搜索的复杂度也越高**，当Beam Size等于1的时候，Beam Search退化为Greedy Search。

Beam Search单条候选序列停止条件细分有两种情况，分别是

- 候选序列解码到停止
- 早停，候选序列得分已经低于已解码完的当前最优序列

![image-20240626195444349](../images/$%7Bfiilename%7D/image-20240626195444349.png)

# 运行

1. 首先配置环境、下载fairseq

```cmd
conda create -n {YOUR_ENV_NAME} python=3.9
conda activate {YOUR_ENV_NAME}
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

2. 下载数据并预处理 注意bash命令需要在git bash中找到对应的目录运行，遇到了下载失败的问题，[解决方案](###bash命令的问题)

```bash
# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..
```

3. 数据下载完成后 使用如下命令进行数据处理，要将$TEXT替换为TEXT的值

```bash
TEXT=examples/translation/iwslt14.tokenized.de-en
fairseq-preprocess --source-lang de --target-lang en --trainpref examples/translation/iwslt14.tokenized.de-en/train2k --validpref examples/translation/iwslt14.tokenized.de-en/valid --testpref examples/translation/iwslt14.tokenized.de-en/test --destdir data-bin/iwslt14.tokenized.de-en --workers 20
```

4. 为了节省时间，只抽取两千数据训练。然后运行下列命令进行训练

注意用set命令设置CUDA

```cmd
set CUDA_VISIBLE_DEVICES=0 
fairseq-train data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --share-decoder-input-output-embed --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --max-tokens 4096 --eval-bleu --eval-bleu-args "{\"beam\": 5, \"max_len_a\": 1.2, \"max_len_b\": 10}" --eval-bleu-detok moses --eval-bleu-remove-bpe --eval-bleu-print-samples --best-checkpoint-metric bleu --maximize-best-checkpoint-metric
```

遇到了Cython组件出错的问题，[解决方案](###Cython问题)

5. 进行推理

```cmd
fairseq-generate data-bin/iwslt14.tokenized.de-en --path checkpoints/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe
```

遇到了Mask类型未定义问题，[解决方案](###Mask类型定义问题)

6. 得到结果

BLEU4为得分

# 问题解决

### bash命令的问题

wget无法使用，通过下文下载：[【Git】解决Git Bash无法使用tree、zip、wget等命令_git bash zip-CSDN博客](https://blog.csdn.net/aidijava/article/details/127114543)

修改prepare-iwslt14.sh，运行后可以正常下载数据。

 6-10 13-17有添加 46有修改

````bash


#!/usr/bin/env bash
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
if [ ! -d "mosesdecoder" ]; then
  git clone https://github.com/moses-smt/mosesdecoder.git
else
  echo "Moses directory already exists. Skipping clone."
fi

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
if [ ! -d "subword-nmt" ]; then
  git clone https://github.com/rsennrich/subword-nmt.git
else
  echo "Subword NMT directory already exists. Skipping clone."
fi

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=iwslt14.tokenized.de-en
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL" -O $GZ
````

### Cython问题

```cmd
# 无法调用numpy
ImportError: numpy.core.multiarray failed to import (auto-generated because you didn't call 'numpy.import_array()' after cimporting numpy; use '<void>numpy._import_array' to disable if you are certain you don't need it).
# cython组件出错
ImportError: Please build Cython components with: python setup.py build_ext --inplace
```

解决方案

```
pip install --upgrade cython
cd path/to/fairseq
python setup.py build_ext --inplace
```

同时在\fairseq\fairseq\data\data_utils_fast.pyx中添加手动引用np

```py
cimport numpy as np
np.import_array()
```

### Mask类型定义问题

关键报错如下：

```
File "D:\anaconda3\envs\transformer\lib\site-packages\fairseq\modules\transformer_layer.py", line 319, in forward
    output = torch._transformer_encoder_layer_fwd(
RuntimeError: Mask Type should be defined
```

解决方法

找到\anaconda3\envs\transformer\lib\site-packages\fairseq\modules\transformer_layer.py

在forward中加入

```py
self.can_use_fastpath=False
```

有点作弊的方法，如果有其他方法不建议使用这个。但是在google colab上可以正常运行，本地可能是环境的问题。

然后又遇到了输出的类型不对的问题

```
AssertionError: expecting key_padding_mask shape of (5, 128), but got torch.Size([128, 5])
```

解决方法 在self.can_use_fastpath=False之后 会转到else中运行，在else中加入

```py
if encoder_padding_mask is not None and encoder_padding_mask.shape != (x.size(1), x.size(0)):
            encoder_padding_mask = encoder_padding_mask.transpose(0, 1)
```

