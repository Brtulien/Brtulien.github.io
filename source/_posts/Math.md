---
title: Math
date: 2023-09-11 22:09:41
tags: [Math, algorithm]
archive: true
categories: algorithm
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

# [0质因数个数 - 蓝桥云课 (lanqiao.cn)](https://www.lanqiao.cn/problems/2155/learning/)

n最大为10^16   

时间复杂度要在O(sqrt(n))

主要是 中间的while循环会直接处理掉所有当前质数的合数 比如i = 2 会一直除以2 直到不能再被2除 使得2 4 6 8 10等合数在后面都不重复计算 而质数就会被留下 比如5 7 等 不会因为除以2而消失 最后会留下来 所以不会漏算 

然后 i < n / i 有效减低了复杂度 

```C++
#include <iostream>
using namespace std;

int main()
{
  // 请在此输入您的代码
    long long n;
    cin>>n;
    long long res = 0;
    for (int i = 2; i <= n / i; i++)
    {
        int num = 0;
        while (n % i == 0)
        {
            n /= i;
            num++;
        }
        if (num > 0)res++;
    }
    if (n > 1)res++;
    cout<<res;
  	return 0;
}
```

# [365. 水壶问题 - 力扣（LeetCode）](https://leetcode.cn/problems/water-and-jug-problem/?envType=daily-question&envId=2024-01-28)

每次水壶只会增加或者减少x 或y的水 只要x + y >= z 找出一对a， b使得**ax+by=z**就可以 那么就需要找出xy的最大公约数

贝祖定理告诉我们，ax+by=z ax+by=z $a*x+b*y=z$ 有解当且仅当 z 是 x y 的最大公约数的倍数

```C++
class Solution {
public:
    bool canMeasureWater(int x, int y, int z) 
    {
        if (x + y < z)return false;
        if (x == 0 || y == 0)return z == 0 || x + y == z;
        return z % gcd(x, y) == 0;
        
    }
};
```

## gcd

```c++
long long gcd(long long a, long long b)
{
	return b == 0: a? gcd(b, a) 
}
```

# [1792. 最大平均通过率 - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-average-pass-ratio/)

优秀的学生要加入那个班级？ 这个就要看 这个学生加入那个班级最好（使得平均通过率增加最多）

就要算平均通过率的增量 按照增量排序

```C++
class Solution {
public:
    double maxAverageRatio(vector<vector<int>>& classes, int extraStudents) 
    {
        priority_queue<tuple<double, int, int>>q;
        for (auto &x: classes)
        {
            int a = x[0], b = x[1];
            double zl = (double) (a + 1) / (b + 1) - (double) a / b;
            q.push({zl, a, b});
        }
        while (extraStudents--)
        {
            auto [_, a, b] = q.top();
            q.pop();
            a++, b++;
            double zl = (double)(a + 1) / (b + 1) - (double)a / b;// 如果再次加到这个班 增量的大小
            q.push({zl, a, b});
        }
        double ans = 0;
        while (q.size())
        {
            auto [_, a, b] = q.top();
            q.pop();
            ans += (double) a / b;
        }
        return ans / classes.size();
    }
};
```

# [1205. 买不到的数目 - AcWing题库](https://www.acwing.com/problem/content/1207/)

## 寻找数学规律的方法

可以把题目先模拟出来找规律 不用手写

找规律可以固定a不变 换b  然后再固定b不变换a

```C++
#include <iostream>
using namespace std;

bool dfs(int m, int p, int q)
{
    if (m == 0)return true;
    
    if (m >= p && dfs(m - p, p, q))return true;
    if (m >= q && dfs(m - q, p, q))return true;
    return false;
}

int main()
{
    int p, q;
    cin>>p >> q;
    int res;
    for (int i = 1; i <= 1000; i++)
    {
        if (!dfs(i, p, q))res = i;
    }
    cout<<res;
}
```

两个数不能凑出的 最大数 要求 两个正整数必须互质才可能有最大不能凑出的数

即当gcd(a, b)>1 的时候无解   互质的时候 两个数最大不能凑出的数为 (a - 1) * (b - 1) - 1

# [1216. 饮料换购 - AcWing题库](https://www.acwing.com/problem/content/1218/)

当还有剩余的瓶盖或者完整的饮料的时候 就要继续循环

```c++
#include <iostream>
using namespace std;
int main()
{
    int n;
    cin>>n;
    int ys = 0, ans = 0;
    while (n)
    {
        ans += n;
        ys += n % 3;
        n /= 3;
        //cout<< n<<" "<<ans<<" "<<ys<<" "<<endl;
    }
    while (n || ys >= 3)
    {
        n = ys / 3;
        ys = ys % 3;
        while (n)
        {
            ans += n;
            ys += n % 3;
            n /= 3;
            //cout<< n<<" "<<ans<<" "<<ys<<" "<<endl;
        }
        
    }
    
    cout<<ans;
}
```

# [1224. 交换瓶子 - AcWing题库](https://www.acwing.com/problem/content/1226/)

置换群

每个长k的环需要k - 1次交换 则求n中有多少个环即可

```C++
#include <iostream>
using namespace std;

int nums[10010];
int vis[10010];
int main()
{
    int n = 0; 
    cin>>n;
    for (int i = 1; i <= n; i++)
    {
        cin>>nums[i];
    }
    int cnt = 0;
    for (int i = 1; i <= n; i++)
    {
        if (!vis[nums[i]])
        {
            cnt++;
            int j = i;
            while (!vis[nums[j]])
            {
                vis[nums[j]] = 1;
                j = nums[j];
            }
        }
    }
    cout<<n - cnt;
}
```

