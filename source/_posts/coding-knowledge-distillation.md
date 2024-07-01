---

title: coding-knowledge_distillation
date: 2024-06-27 09:27:35
tags: [Knowledge_Distillation, deep learning, coding]
categories: deep learning
---

## [fairseq的命令行参数](https://fairseq.readthedocs.io/en/latest/command_line_tools.html)

本次实验基于fairseq，目的是熟悉fairseq的命令行参数、实现损失函数

# 问题

## 1.1 

Q: fairseq中，inference时默认是使用test数据集，如何改成train数据集：

* 方法一

直接修改命令行参数

```bash
fairseq-generate data-bin --gen-subset train --path model.pt
```

参考

![image-20240627151341390](../images/$%7Bfiilename%7D/image-20240627151341390.png)

* 方法二 

修改源码

fairseq的inference一般是在fairseq/fairseq_cli/generate.py中

从入口函数\__main__开始找 到cli_main

cli_main中主要是定义命令行参数和调用main，传入参数args。

```py
def cli_main():
    parser = options.get_generation_parser()
    # TODO: replace this workaround with refactoring of `AudioPretraining`
    parser.add_argument(
        "--arch",
        "-a",
        metavar="ARCH",
        default="wav2vec2",
        help="Model architecture. For constructing tasks that rely on "
        "model args (e.g. `AudioPretraining`)",
    )
    args = options.parse_args_and_arch(parser)
    main(args)
```



将传入main的参数args转为DictConfig对象，这意味着在命令行参数解析和转换过程中，**所有命令行选项都会映射到cfg这个配置对象上**。

```py
def main(cfg: DictConfig):
```

所以说虽然main中cfg的某些参数在cli_main中没有定义，但是可以在通过命令行参数进行推理的时候，进行添加，比如：

```bash
python generate.py --gen-subset train
```

这样就可以定义。

具体修改在106行

```py
# loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
# 在这里加上
cfg.dataset.gen_subset = 'train'
task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)
```

## 1.2

Q: fairseq中，inference生成的结果还包含其它信息，不能直接添加到原数据集中，如何处理为仅含源句/目标句

* 方法一

可以在推理之后再用python文件操作处理

* 方法二

通过观察同文件中255行之后的代码可以发现，其中控制了输出的格式。

```py
            if not cfg.common_eval.quiet:
                if src_dict is not None:
                    print("S-{}\t{}".format(sample_id, src_str), file=output_file)
                if has_target:
                    print("T-{}\t{}".format(sample_id, target_str), file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][: cfg.generation.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo["tokens"].int().cpu(),
                    src_str=src_str,
                    alignment=hypo["alignment"],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=cfg.common_eval.post_process,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not cfg.common_eval.quiet:
                    score = hypo["score"] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        "H-{}\t{}\t{}".format(sample_id, score, hypo_str),
                        file=output_file,
                    )
                    # detokenized hypothesis
                    print(
                        "D-{}\t{}\t{}".format(sample_id, score, detok_hypo_str),
                        file=output_file,
                    )
                    print(
                        "P-{}\t{}".format(
                            sample_id,
                            " ".join(
                                map(
                                    lambda x: "{:.4f}".format(x),
                                    # convert from base e to base 2
                                    hypo["positional_scores"]
                                    .div_(math.log(2))
                                    .tolist(),
                                )
                            ),
                        ),
                        file=output_file,
                    )

                    if cfg.generation.print_alignment == "hard":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [
                                        "{}-{}".format(src_idx, tgt_idx)
                                        for src_idx, tgt_idx in alignment
                                    ]
                                ),
                            ),
                            file=output_file,
                        )
                    if cfg.generation.print_alignment == "soft":
                        print(
                            "A-{}\t{}".format(
                                sample_id,
                                " ".join(
                                    [",".join(src_probs) for src_probs in alignment]
                                ),
                            ),
                            file=output_file,
                        )

                    if cfg.generation.print_step:
                        print(
                            "I-{}\t{}".format(sample_id, hypo["steps"]),
                            file=output_file,
                        )

                    if cfg.generation.retain_iter_history:
                        for step, h in enumerate(hypo["history"]):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h["tokens"].int().cpu(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                "E-{}_{}\t{}".format(sample_id, step, h_str),
                                file=output_file,
                            )
```

```
S-0	源句。
T-0	目标句。
H-0	-0.123456789	生成的句子。
D-0	-0.123456789	生成的句子。
P-0	-0.1234 -0.2345 -0.3456  得分
A-0	0-0 1-1 2-2  对齐信息
```



因此可以通过修改命令行参数或者直接修改源码，控制输出为原句/目标句。(S T)

```bash
fairseq-generate --quiet False
```

<img src="../images/$%7Bfiilename%7D/image-20240627172002354.png" alt="image-20240627172002354" style="zoom:50%;" />

<img src="../images/$%7Bfiilename%7D/image-20240627172054620.png" alt="image-20240627172054620" style="zoom:50%;" />

但是观察代码发现，当输出S T的时候，设置命令行参数为quiet，这会导致H D P A 也同时被输出；--print-alignment不论设置soft还是hard 都会输出额外的信息；而其他的一些参数如 --retain_iter_history --print_step等 默认值就为不输出 所以无需设置。

因此直接修改命令行参数可能无法达到预期效果，需要修改代码。

## 1.3

使用的是第一次代码训练中的transformer的训练结果。

* 获取soft-target

```bash
fairseq-generate data-bin/iwslt14.tokenized.de-en --path checkpoints/checkpoint_best.pt --batch-size 128 --beam 5 --remove-bpe --gen-subset train --source-lang de --target-lang en --results-path distillation_output
```

* 训练学生模型

```bash
fairseq-train data-bin/iwslt14.tokenized.de-en --arch transformer_iwslt_de_en --share-decoder-input-output-embed --optimizer adam --adam-betas "(0.9,0.98)" --clip-norm 0.0 --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 --dropout 0.3 --weight-decay 0.0001 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --source-lang de --target-lang en  --save-dir checkpoints/student
```

* 测试 生成翻译

这里batch-size128太大了，gpu报错现存不足，改成64即可。改的太小也不行，会出现未定义错误，可能最小就是64。

```bash
fairseq-generate data-bin/iwslt14.tokenized.de-en --path checkpoints/student/checkpoint_best.pt --batch-size 64 --beam 5 --remove-bpe --gen-subset test --source-lang de --target-lang en --results-path test_output
```

训练完成后数据保存在test_output中。格式如下（没有按照上一问修改，确实有很多东西）。有用的就是生成句子和参考句子。用他们计算BLEU值。

```
S-260	ein laser ist da anders .
T-260	now a laser is different .
H-260	-3.3165059089660645	. . . .
D-260	-3.3165059089660645	. . . .
P-260	-4.3481 -2.7208 -2.7019 -2.6929 -4.1188
```

本地训练有点困难，训练的epoch不多，训练数据也只截取了一部分，可能不太准确。但是有提高一点点。

## 2.1

Q: KL散度是非常重要的函数，请学习其概念并说明其定义

公式:

$D_{KL}(P||Q)=\sum_{x\in X}P(x)log\frac{p(x)}{Q(x)}$

KL散度是用来衡量两个概率分布之间差异的一种度量方法。就是计算p，q之间的相对熵。按信息论的说法就是，某一个事件在系统q中的信息量和系统p中的信息量的差值，对这个差值求期望（平均）。

$D_{KL}(P||Q)=\sum_{i=1}^{m}p_i(f_Q(q_i)-f_P(p_i))$

由这个公式可以推导出最上面的KL散度的公式，这个公式意义可能好理解一些，就是事件$q_i$在P、Q两个概率分布中的差异。

衡量分布Q生成的数据相对于分布P生成的数据的额外信息量，也就是使用Q做近似分布时失去的信息。

可以用作**损失函数**或者**相似性函数**。

## 2.2

--arch是Fairseq的一个命令行参数，用于指定训练模型时使用的模型架构。使用方法如下：

```
 --arch transformer_iwslt_de_en
```

在Fairseq的代码中，模型架构通常使用@register_model_architecture来注册，模型用@register_model来注册。

在训练时，通过--arch来获取所需的教师模型。比如本次使用transformer进行训练，就用如上的代码获取transformer模型。

## 2.3

调参是训练过程中的重要环节，但是有时候需要保持某些参数不变，这时候就要冻结模型参数。一般来说冻结模型参数是为了保存已有的特征提取能力，防止因为数据改变倒是模型性能下降。

可以用下列代码冻结全部参数。也可以设置条件冻结部分参数。

```py
for param in model.parameters():
        param.requires_grad = False
```

获取教师模型的概率分布

```py
def get_teacher_probs(teacher_model, input_data):
    with torch.no_grad():  
        logits = teacher_model(input_data)   
        probs = F.softmax(logits, dim=-1)  
    return probs
```

KL散度

```py
def distillation_loss(student_logits, teacher_probs, temperature=1.0):
    log_student_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_probs / temperature, dim=-1)
    loss = F.kl_div(log_student_probs, teacher_probs, reduction='batchmean') * (temperature ** 2)
    return loss
```



