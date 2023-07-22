---
title: python速成笔记
date: 2023-07-22 23:19:03
tags: study
---


# python速成的一些学习笔记

* pow函数可以实现快速幂
* 对Unicode的字符使用函数ord可以将其转换成对应的Unicode编码 逆向转换使用chr
* 字符串居然也可以用加和乘的运算 但是好像不能用减？ 
* 字符串和列表都有方便的子串/元素检测 in 如 a in str
* 字符串与列表的转换
```
num = list(range(65,70))            
lis = [chr(x) for x in num]           
print(lis)                  
s = ''.join(lis)                    
print(s)            
``` 
* 二维[数组]
```
via = [[0]*3 for _ in range(3)]                    
print(via)              
via[0][0] = 1               
print(via)              
```
* 使用NumPy建立多维数组和访问
```
import numpy as np      

#容量为3 未初始化       
lis = np.empty(3)       
print(lis)      
#3*3 初始化为0      
lis = np.zeros(3*3)     
print(lis)      
#整数数组       
lis = np.zeros(3*3, dtype=int)      
print(lis.shape)        
#获取数组最大值         
np.max(lis)     
#展平       
lis.flatten()       
#对每行排序 返回排序结果
np.sort(lis, axis=1)
#行方向原地排序
lis.sort(axis=1)
```
* 输入输出
```
#浮点数输出
pi = 3.1415926
print("%.4f" % pi)
"%.4f - %8f = %d" % (pi, 0.1416, 3)

#输入
s = input()
a = s.split()
print(a)
a = [int(x) for x in a]
#或者
a = [int(x) for x in input().split()]
#固定输入
u, v, w = [int(x) for x in input().split()]
```

* 内置容器
```
# 字典
dic = {}
dic = {"key", "value"}
dic1 = {chr(i): i for i in range(65, 69)}
# 翻转
dic1 = {dic1[k]: k for k in dic1}
print(dic1)
# 要先判断才能打印
if 'b' in dic1:
    print(dic1['b'])
else:
    dic1['b'] = 98
```