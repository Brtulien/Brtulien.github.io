---
title: LCA
date: 2023-08-22 12:24:01
tags: [algorithm, lca, Tree]
---

·最近公共祖先

```python
root = 1
num = 0
dep = [0] * 1000010
f = [[0] * 21 for _ in range(1000001)]
head = [-1] * 10000010

class Edge:
    def __init__(self, to, next):
		self.to = to
         self.next = next
            
def addedge(from_,to):
	global num
    num += 1
    e[num] = Edge(to, head[from_])
    head[from_] = num
    
def dfs(v, father):
	dep[v] = dep[father] + 1
    f[v][0] = father
    for i in range(1, 21):
        f[v][i] = f[f[v][i-1]][i-1]
        
    i = head[v]
    while i != -1:
        p1 = e[i].to
        if p1 == father:
            i = e[i].next
            continue
        dfs(p1,v)
        i = e[i].next

def lca(x, y):
	if dep[x] < dep[y]:
        x, y = y, x
    for i in range(20, -1, -1):
		if dep[f[x][i]] >= dep[y]:
            x = f[x][i]
         if x == y:
			return x
        
    for i in range(20, -1, -1):
        if f[x][i] != f[y][i]:
            x = f[x][i]
            y = f[y][i]
            
    return f[x][0]


n, m, root = map(int, input().split())
e = [Edge(0, 0) for _ in range(1000001)]
for _ in range(n-1):
    u, v = map(int, input().split())
    addedge(u, v)
    addedge(v, u)

# 建立 LCA 预处理
dfs(root, 0)

# 查询 LCA
for _ in range(m):
    x, y = map(int, input().split())
    result = lca(x, y)
    print(result)

```

