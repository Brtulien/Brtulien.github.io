---
title: KMP
date: 2023-10-11 23:43:56
tags: algorithm
---

j

将模式串指针j的回溯 看作整个模式串向后移动来匹配主串

当i j不匹配的时候有两种做法，第一是模式串移动到当前串的后面（对应的其实是next数组移动到0的位置的时候 也就是全不匹配的时候）

```
a b a b a b a 
a b a b a c b 
移动后
a b a b a b a 
          a b a b a c b
```

第二种 移动到下一个公共子串的位置

```
a b a b a b a 
a b a b a c b 
移动后
a b a b a b a 
    a b a b a c b
```

即主串的后缀集合与模式串的前缀集合有交集的时候

j指针回溯的位置是A后缀和B前缀交集里最长的元素（这样不会遗漏）最长元素的长度就是j回溯的位置

匹配失败 但是A和B串存在一段相同的子串 j回溯的位置只与B有关（A、B相同子串的前后缀 其实就是B子串的前后缀） 因此可以先求出B放入next数组

next[i]表示B[1]~B[i]最长公共前后缀的长度



匹配步骤

i, j初始化为0

1.如果A[i + 1] == B[j + 1] i++,j++

2.如果不相等 不断回溯j到next[j] 直到A[i + 1] == B[j + 1]   或者j回溯到next[0] = -1了 此时直接让i++（）意思是B移到A后面

3.j = m 匹配成功输出位置 



构建next数组  

求B[1]~B[i]最长公共后缀长度

如果匹配 next[i] = j + 1  (j 表示B串前缀的指针 也就是当前字符匹配之前的最长公共前后缀长度 匹配成功就+1)

匹配不成功 回溯j指针j = next[j]直到成功



```C++
#include <iostream>
using namespace std;
void getnext(string p, int next[])
{
    int p_len = p.size();
    int i = 0;
    int j = -1;
    next[0] = -1;
    
    while(i < p_len)
    {
        if (j == -1 || p[i] == p[j])
        {
            i++;
            j++;
            next[i] = j
		}
        else
           	j = next[j]
    }
}
void KMP(string s, string p, int next[])
{
    getnext(p,next);
    
    int s_len = s.size();
    int p_len = p.size();
    int i = 0;
    int j = 0;
    
    while (i < s_len && j < p_len)
    {
        if (j == -1 || s[i] == p[j])
        {
            i++;
            j++;
        }
        else
            j = next[j]
    }
    if (j == p_len)return i - j;
    return -1;
}
```

