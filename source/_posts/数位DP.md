---
title: 数位DP
date: 2024-01-17 11:06:18
tags: [algorithm, Ans, DP]
---



# 基本思想

```
以第一题为例比如n = 1234 要找1~1234的数
数位DP是 一位一位地去填 看能不能填上这个数
比如1 _ _ _ 第一位填1 那么第二位就只能填 0 1 2 中的数 （否则就会超出1234）
1 1 _ _ 第二位填1 那么第三位 第四位都可以随意地填0~9的任何值 因为不论怎么填都不会超过1234 
```

# 数位DP模板 

```C++
function<int(int, int, bool, bool)>f = [&] (int i, int mask, bool isLimit, bool isNum) -> int{}
```

## 参数含义

i代表现在走到第几位数

mask 根据题目而定 也可能没有 也可能是cnt等 主要用于记录题目的条件 比如第一题 用cnt 记录1的个数 第二题是用mask记录是否有重复的数字

isLimit 看是否有受到限制 这涉及到[数位DP的思想](#基本思想)：

则isLimit就是判断这一位能填写的上下界（主要是上界up）如果不受限制 那就是9 受限制就是 n在这一位的值s[i]

isNum 主要是用来判断前导0的情况 如果前面有0 就不是一个数了 如果题目不含前导0则不需要

## 步骤

首先把n转换为字符串 然后求出长度m 初始化记忆化数组

写function函数

退出条件 一般是i == m 返回 什么？

记忆化 一般是！isLimit && dp[i] [cnt/mask] != 0/-1 && isNum 直接return

res = 0 用于记录答案

求出最大最小值 up 和 low

循环for d in range(low, up) 在循环内递归

if (d == 1 // mask>>d&1 == 0) 选择的这个d 如果满足题目条件

​	就递归 res += f(i + 1, ???, isLimit && d == up, ???) 注意更新状态 

最后 还需要写入记忆化 if(!isLimit && isNum)dp[i] [cnt/ mask] = res

return res

注意初始条件

# [233. 数字 1 的个数 - 力扣（LeetCode）](https://leetcode.cn/problems/number-of-digit-one/solutions/1750339/by-endlesscheng-h9ua/)

```C++
class Solution {
public:
    int countDigitOne(int n) {
    	auto s = to_string(n);
    	int m = s.size(), dp[m][m];
    	memset(dp, -1, sizeof(dp));
    	
    	function<int(int, int, bool)>f = [&](int i, int cnt, bool isLimit) -> int
        {
            if (i == m)
                return cnt;
            if (!isLimit && dp[i][cnt] >= 0) 
                return dp[i][cnt];
            int res = 0; 
            int up = isLimit ? s[i] - '0': 9;
            for (int d = 0; d <= up; d++)
            {
                res += f(i + 1, cnt + (d == 1), isLimit && d == up);
            }
            if (!isLimit) dp[i][cnt] = res;
            return res;
        };
        return f(0, 0, true);// 一开始需要限制
    }
};

```

# [2376. 统计特殊整数 - 力扣（LeetCode）](https://leetcode.cn/problems/count-special-integers/description/)

```C++
class Solution {
public:
    int countSpecialNumbers(int n) 
    {
     	string s = to_string(n);
        int m = s.size(), dp[m][1<<m];
        memset(dp, -1, sizeof(dp));
        
        function<int(int, int, bool, bool)>f = [&](int i, int mask, int isLimit, int isNum) -> int
        {
            if (i == m)
                return isNum;
            if (!isLimit && isNum && dp[i][mask] != -1)
                return dp[i][mask];
            int res = 0;
            if (!isNum)
                res = f(i + 1, mask, false, false);
            int up = isLimit ? s[i] - '0': 9;
            int low = isNum ? 1: 0;
            for (int d = low; d <= up; d++)
            {
                if ((mask>>d & 1) == 0)
                    res += f(i + 1, mask | (1<<d), isLimit && up == d, true);
            }
            if (!isLimit && isNum)
                dp[i][mask] = res;
            return res;
        };
        return f(0, 0, true, false);
    }
};
```

# [3007. 价值和小于等于 K 的最大数字 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-number-that-sum-of-the-prices-is-less-than-or-equal-to-k/)

二分+数位DP

```C++
typedef long long ll;
class Solution {
public:
    ll CntDigitOne(ll n, int x)
    {
        int m = 64 - __builtin_clzll(n);
        ll dp[m][m + 1];
        memset(dp, -1, sizeof(dp));

        function<ll(int, int, bool)>f = [&](int i, int cnt, int isLimit) -> ll
        {
            if (i < 0)
                return cnt;
            if (!isLimit && dp[i][cnt] >= 0)
                return dp[i][cnt];
            ll res = 0;
            int up = isLimit? n>>i & 1: 1;
            for (int d = 0; d <= up; d++)
            {
                res += f(i - 1, cnt + (d == 1 && ((i + 1) % x == 0)), isLimit && up == d);
            }
            if (!isLimit)dp[i][cnt] = res;
            return res;
        };
        return f(m - 1, 0, true);
    }
    long long findMaximumNumber(long long k, int x) 
    {
        ll left = 1, right = (k + 1) << x;
        ll res = 0;
        while (left <= right)
        {
            ll mid = (left + right) >> 1;
            ll s = CntDigitOne(mid, x);
            if (s <= k)
            {
                left = mid + 1;
                res = mid;
            }
            else right = mid - 1;
            
        }
        return res;
    }
};
```

