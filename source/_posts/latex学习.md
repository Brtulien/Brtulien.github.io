---
title: latex学习
date: 2023-08-03 06:54:55
tags: [study, latex]
archive: true
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





## 正文部分

* \textbf{}加粗
* \textit{}斜体
* underline 下划线



## 章节

```latex
\documentclass[UTF8]{ctexart}

\begin{document} //正文

\part{A} // 该部分标题为A 并且开始一个部分 直到下一个部分

\chapter{B} //该章节标题为B 并且开始一个部分 直到下一个部分

\section{C} //该章节标题为C 并且开始一个部分 直到下一个部分

\subsection{D} //创建子章节

\subsubsection //.

\end{document}
```

## 图片

```latex
\documentclass[UTF8]{ctexart}
\usepackage{graphicx} //引入宏包
\begin{document}

\begin{figure}
\centering // 居中
\includegraphics[width=0.5\textwidth]{head}
\caption{...} //图片标题
\end{figure}

\end{document}
```

## 列表

```latex
\documentclass[UTF8]{ctexart}
\begin{document}

//无序列表环境为itemize
\begin{enumerate}
\item 列表项1
\item 列表项2
\end{enumerate}

\end{document}
```

## 公式

```latex
\documentclass[UTF8]{ctexart}

\begin{document}

// 行内公式
AAAAA $E = mc^2$
// 行间公式
// 使用equation环境
\begin{equation}
E = mc^2
\end{equation}
// 或者用\[ 和 \]
\[
E = mc^2
\]
\end{document}
```

## 表 格

```latex
\documentclass[UTF8]{ctexart}
\begin{document}

\begin{table}// 需要标题和居中的时候要把表格放在table环境中

\center

\begin{tabular}{c c c}  // c | c c代表表格共三列 每列内容都居中对齐 用l表示左对齐 r右 添加竖线代表竖边框 水平边框通过\hline添加 每格之间用&隔开 每行之间用\\隔开
// 把c改成p{2cm}自定义列宽
\hline
单元格1 & 单元格2 & 单元格3 \\      
\hline //双横线
\hline
单元格4 & 单元格5 & 单元格6 \\
单元格7 & 单元格8 & 单元格9 
\end{tabular}

\caption{title}

\end{table}

\end{document}
```

