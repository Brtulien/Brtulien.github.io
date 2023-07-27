---
title: python语法
subtitle: 你笑什么啊~不许笑！我才刚看完语法啊，呜呜
date: 2023-07-23 22:03:27
tags: study python
---
# python语法注意事项
~~啊，受不了了，写了几个题不是运错就是编错，还有一堆奇奇怪怪的报错，痛苦.jpg。~~    
已经遇见未来不短的时间内会不断报错并且看不懂然后写在这里了，希望这篇可以早日停更。

* n = int(input())
* 控制不住if后面加()，哭，剁手
* xmuoj不支持类型注解，呜呜
* 没有while(n--)的操作了，用while会忘记--，死循环了，好蠢
* range的范围
* 读入二维列表 (老是忘记lis的append，想半天)
```
lis = []
for i in range(n):
    d = list(map(int,input().split()))
    lis.append()
```
* 浮点数确定精度输出 print("%.1f" % n) 注意是双引号并且没有逗号
* dx = [1,0,-1,0] ~~这里是今晚最佳~~ 加了个list怎么找也找不出来，list dx[]->笑死

* 全排列 选哪个？ 标记选过的不再选
```
def permute(sol,num,on_path):
    if len(sol) == len(num):
        for i in range(n):
            print(sol[i], "", end='')
        print()
        return
    else:
        for i in range(n):
            if on_path[i] == 1:
                continue
            else:
                on_path[i] = 1
                permute(sol+[num[i]],num,on_path)
                on_path[i] = 0
                

n = int(input())
sol = []
num = [i for i in range(1,n+1)]
on_path = [0 for i in range(n)]
permute(sol,num,onpath]
```

* 字典的使用方法  记得一定要先判断是否存在 ~~呜呜，明明之前才记过，写的时候又忘记了，泰蠢辣~~
```
st = input()

d = dict()
for ch in st:
    if ch in d:
        d[ch] += 1
    else:
        d[ch] = 1
```
* 上面的题是用来求只出现一次的字符的 难过 在c++知道用count 在python就变蠢力
* python线性筛 芜湖~
```
N = 100010
st = [False for _ in range(N)]
prime = [0 for _ in range(N)]

def get_prime(n):
    cnt = 0
    for i in range(2,n + 1):
        if not st[i]:
            prime[cnt] = i
            cnt += 1
    j = 0
    while prime[j] < n // i:
        st[prime[j] * i] = True
        if i % prime[j] == 0:
            break
        j += 1
```
* range是左开右闭
* 我超 在函数里面用全局变量的时候要加global 
* python归并
```
n = int(input(n))
nums = list(map(int,input()))
temp = [0 for _ in range(n)]

merge_sort(l, r):
    if l >= r:
        return 
    mid = (l + r) >> 1
    merge_sort(l, mid)
    merge_sort(mid + 1, r)
    i, j, k = l, mid + 1, 0
    while i <= mid and j <= r:
        if nums[i] <= nums[j]:
            temp[k] = nums[i]
            k += 1
            i += 1
        else:
            temp[k] = nums[j]
            k += 1
            j += 1
    while i <= mid:
        temp[k] = nums[i]
        k += 1
        i += 1
    while j <= r:
        temp[k] = nums[j]
        k += 1
        j += 1
    
    j = 0
    for i in range(l, r + 1):
        nums[i] = temp[j]
        j += 1
        
        
merge_sort(0,n - 1)# n-1!!
```

* python手动增加递归的层数   
import sys      
sys.setrecursionlimit(5000)