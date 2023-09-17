---
title: Math
date: 2023-09-11 22:09:41
tags: [Math, algorithm]
---

# 线性筛

```python
"""
如果i是primes[j]的倍数，跳出循环是因为在线性筛的过程中，我们的目标是找到小于等于n的所有素数，并且要保证每个合数只被标记一次。因此，在内层循环中，当i是primes[j]的倍数时，我们不需要再继续考虑primes[j] * i及其之后的倍数了，因为它们已经在之前的迭代中被标记过了。

举个例子来说明：

假设我们正在处理i=10，而primes[j]=2，也就是说10是2的倍数。那么在这一轮迭代中，我们会标记10、20、30、40、...等等所有10的倍数为非素数。但实际上，这些数在之前已经被标记过了，因为它们分别是2、4、6、8、...等等的倍数，而这些倍数在处理2的时候已经被标记过了。所以，为了避免重复标记，当i是primes[j]的倍数时，我们可以直接跳出内层循环，不再处理这个数及其后续的倍数。
"""

def linear_sieve(n):
    is_prime = [True] * (n + 1)  # 初始化一个布尔数组，标记每个数是否为素数
    primes = []  # 存储素数的列表

    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)  # i是素数
        for j in range(len(primes)):
            # 将已知素数primes[j]与当前数i相乘，标记其倍数为非素数
            if primes[j] * i > n:
                break
            is_prime[primes[j] * i] = False
            if i % primes[j] == 0:  # 如果i是primes[j]的倍数，跳出循环
                break

    return primes

n = 30  # 你可以设置需要查找素数的上限
primes = linear_sieve(n)
print("小于等于", n, "的素数：", primes)

```

