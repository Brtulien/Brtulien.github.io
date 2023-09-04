---
title: python速成笔记
date: 2023-07-22 23:19:03
tags: [study, python]
---

# python速成的一些学习笔记

什么？！你说python速成？->[[Python 速成 - OI Wiki](https://oi.wiki/lang/python/)]

* pow函数可以实现快速幂
* 对Unicode的字符使用函数ord可以将其转换成对应的Unicode编码 逆向转换使用chr
* 字符串居然也可以用加和乘的运算 但是好像不能用减？ 
* 字符串和列表都有方便的子串/元素检测 in 如 a in str
* 字符串与列表的转换
```python
num = list(range(65,70))            
lis = [chr(x) for x in num]           
print(lis)                  
s = ''.join(lis)                    
print(s)            
```
* 二维[数组]
```python
via = [[0]*3 for _ in range(3)]                    
print(via)              
via[0][0] = 1               
print(via)              
```
* 使用NumPy建立多维数组和访问
```python
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
```python
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
```python
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



# 更新！学了两个月之后再看一遍

* format() 输出浮点数

  ```python
  pi = 3.14159265359
  formatted_pi = "圆周率的近似值是 {:.2f}".format(pi)
  print(formatted_pi)
  :.2f 是一个格式规范，它指定了要显示小数点后两位的浮点数
  
  x = 10
  y = 20
  result = "x 的值是 {}，y 的值是 {}，它们的和是 {}。".format(x, y, x + y)
  print(result)
  
  name = "Alice"
  age = 30
  message = f"我的名字是 {name}，年龄是 {age}。"
  print(message)
  ```

* 输入输出

  ```python
  u, v, w = [int(x) for x in input().split()]
  
  # 二维数组
  mat = [[int(x) for x in input().split()] for i in range(N)]
  
  >>> N = 4; mat = [[int(x) for x in input().split()] for i in range(N)]
  1 3 3 
  1 4 1 
  2 3 4 
  3 4 1 
  >>> mat  # 先按行读入二维数组
  [[1, 3, 3], [1, 4, 1], [2, 3, 4], [3, 4, 1]]
  >>> u, v, w = map(list, zip(*mat))   
  # *将 mat 解包得到里层的多个列表
  # zip() 将多个列表中对应元素聚合成元组，得到一个迭代器
  ## 内置函数 zip() 可以将多个等长序列中的对应元素拼接在「元组」内，得到新序列
  # map(list, iterable) 将序列中的元素（这里为元组）转成列表
  >>> print(u, v, w)  # 直接将 map() 得到的迭代器拆包，分别赋值给 u, v, w
  [1, 1, 2, 3] [3, 4, 3, 4] [3, 1, 4, 1]
  ```

* 内置容器

  ```python
  # dict
  dic = {chr(i): i for i in range(65,91)}
  dic = dict(zip([chr(i) for i in range(65,91)], range(65, 91)))
  # 键值对逆转
  dic = {dic[k]: k for k in dic}
  dic = {v: k for k, v in dic.items()}  # 和上行作用相同，dic.items() 以元组存放单个键值对
  dic = {k: v for k, v in sorted(dic.items(), key=lambda x:-x[1])}  # 字典按值逆排序，用到了 lambda 表达式
  ```

* 装饰器

  lru_cache转记忆化