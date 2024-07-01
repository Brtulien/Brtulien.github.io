---
title: lqb
date: 2024-02-06 16:25:42
tags: Ans
archive: true
categories: algorithm
---

# [1.平方差 - 蓝桥云课 (lanqiao.cn)](https://www.lanqiao.cn/problems/3213/learning/?subject_code=1&group_code=5&match_num=14&match_flow=1&origin=cup)

```C++
#include<iostream>
#include<vector>

using namespace std;

bool cmp(vector<int> &A, vector<int> &B) {
    if(A.size() != B.size()) return A.size() > B.size();

    for(int i = A.size() - 1; i >= 0; i--) {
        if(A[i] != B[i]) return A[i] > B[i];
    }

    return true;
}

vector<int> sub(vector<int> &A, vector<int> &B) {
    vector<int> C;

    for(int i = 0, t = 0; i < A.size(); i++) {
        t = A[i] - t;
        if(i < B.size()) t -= B[i];
        C.push_back((t + 10) % 10);
        if(t < 0) t = 1;
        else t = 0;
    }

    while(C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}

vector<int> mul(vector<int> &A, vector<int> &B) {
    vector<int> C(A.size() + B.size());

    for(int i = 0; i < A.size(); i++) 
        for(int j = 0; j < B.size(); j++) 
            C[i + j] += A[i] * B[j];

    for(int i = 0, t = 0; i < C.size(); i++) {
        t += C[i];
        C[i] = t % 10;
        t /= 10;
    }

    while(C.size() > 1 && C.back() == 0) C.pop_back();

    return C;
}

int main() {
    vector<int> A, B, C;
    string aa, bb, a, b;

    cin>>aa>>bb;

    if(aa[0] == '-') 
        for(int i = 1; i < aa.size(); i++) a += aa[i];
    else a = aa;
    if(bb[0] == '-') 
        for(int i = 1; i < bb.size(); i++) b += bb[i];
    else b = bb;

    for(int i = a.size() - 1; i >= 0; i--) A.push_back(a[i] - '0');
    for(int i = b.size() - 1; i >= 0; i--) B.push_back(b[i] - '0');

    A = mul(A, A);
    B = mul(B, B);

    if(cmp(A, B)) C = sub(A, B);
    else {
        cout<<'-';
        C = sub(B, A);
    }

    for(int i = C.size() - 1; i >= 0; i--) {
        cout<<C[i];
    }
    cout<<endl;
    
    return 0;
}
```

# [1210. 连号区间数 - AcWing题库](https://www.acwing.com/problem/content/description/1212/)

有点偏技巧 关键要找到规律 怎么样求出一个区间是否连号   只要一个区间的最大值和最小值的差 等于 区间的长度就是连号的区间   比如3 2 4    4 - 2 = 2 区间长度（下标分别为0 1 2 ）为2 - 0 = 2   

因为当排好序之后 连号数列为minn x1 x2... maxn 则 maxn - minn = 区间长度

找到规律之后直接枚举 左右区间

```c++
#include <iostream>
using namespace std;
int nums[100010];
int main()
{
    int n;
    cin>>n;
    for (int i = 0; i < n; i++)
    {
        cin>>nums[i];
    }
    int maxn = nums[0], minn = nums[0];
    int cnt = 0;
    for (int l = 0; l < n; l++)
    {
        maxn = minn = nums[l];
        for (int r = l; r < n; r++)
        {
            maxn = max(maxn, nums[r]);
            minn = min(minn, nums[r]);
            if (maxn - minn == r - l)cnt++;
        }
    }
    cout<<cnt;
}
```

