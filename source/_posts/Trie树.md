---
title: Trie树
date: 2023-11-16 23:58:59
tags: algorithm
archive: true
categories: algorithm
---

# [2935. 找出强数对的最大异或值 II - 力扣（LeetCode）](https://leetcode.cn/problems/maximum-strong-pair-xor-ii/submissions/482713400/)

```py
class Node():
    __slots__ = 'children', 'cnt'

    def __init__(self):
        self.children = [None, None]
        self.cnt = 0

class Trie():
    HIGH_BIT = 19

    def __init__(self):
        self.root = Node()
    
    def insert(self, val: int) -> None:
        cur = self.root
        for i in range(Trie.HIGH_BIT, -1, -1):
            bit = (val >> i) & 1
            if cur.children[bit] is None:
                cur.children[bit] = Node()
            cur = cur.children[bit]
            cur.cnt += 1
        return cur

    def remove(self, val: int) -> None:
        cur = self.root
        for i in range(Trie.HIGH_BIT, -1, -1):
            cur = cur.children[(val >> i) & 1]
            cur.cnt -= 1
        return cur

    def max_xor(self, val: int) -> int:
        cur = self.root
        ans = 0
        for i in range(Trie.HIGH_BIT, -1, -1):
            bit = (val >> i) & 1
            if cur.children[bit ^ 1] and cur.children[bit ^ 1].cnt:
                ans |= 1 << i
                bit ^= 1
            cur = cur.children[bit]
        return ans

class Solution:
    def maximumStrongPairXor(self, nums: List[int]) -> int:
        root = Trie()        
        nums.sort()
        n = len(nums)
        ret = left = 0
        for y in nums:
            root.insert(y)
            while nums[left] * 2 < y:
                root.remove(nums[left])
                left += 1
            ret = max(ret, root.max_xor(y))            
        return ret
```



