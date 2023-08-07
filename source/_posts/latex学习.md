---
title: latex学习
date: 2023-08-03 06:54:55
tags: [study, latex]
---

## 格式

* 首先\documentclass[UTF8]{ctexart}

  ....(宏包)

  \begin{document}

  ......

  \end{document}

* 行内要用$$括起来 行间要用$$$$括起来

* 公式对齐 不同的环境语法不同 eqnarray是&=&；align是&=

* 分段函数  cases环境  使用& 对齐  \\\换行

* 矩阵 array begin后面加{ccc}表示格式  需要$$$$和自己加括号

* 要表示带省略号的矩阵 用pmatrix环境

* 表格 数字表格即矩阵加|，带汉字的要使用tabular环境。表格的横线为\hline

* 插图 \includegraphics[scale=*] [width=*][height=*]{.png}

  或使用\figure环境

  \begin{figure}[H]

   \centering

   % Requires \usepackage{graphicx}

   \includegraphics[width=12pt]{.png}\\

   \caption{图1}

  \end{figure}
