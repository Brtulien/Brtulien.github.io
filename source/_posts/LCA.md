---
title: LCA
date: 2023-08-22 12:24:01
tags: [algorithm, lca, Tree]
---

# 最近公共祖先 数组

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

```C++
void dfs(int x, int father)
{
	fa[x] = father;
	de[x] = de[father] + 1;
	for (int i = head[x]; i; i = edge[i].next)
	{
		if (edge[i].to != father)
		{
			dfs(edge[i].to, x);
		}
	}
}

int lca(int x, int y)
{
	while (x != y)
	{
		if (de[x] >= de[y])
		{
			x = fa[x];
		}
		else
			y = fa[y];
	}
	return x;
}
```

# 当是链表形式的时候 较为简单

## [236. 二叉树的最近公共祖先 - 力扣（LeetCode）](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/description/)

```C++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) 
    {
        if (root == nullptr || root == p || root == q)
            return root;
        
        TreeNode *left = lowestCommonAncestor(root->left, p, q);
        TreeNode *right = lowestCommonAncestor(root->right, p, q);

        if (left && right)
            return root;
        return left ? left : right;
    }
};
```

## [2096. 从二叉树一个节点到另一个节点每一步的方向 - 力扣（LeetCode）](https://leetcode.cn/problems/step-by-step-directions-from-a-binary-tree-node-to-another/description/)

从st到end的路程中 必定会经过两个点的最近公共祖先 

1如果两个点没关系  那就先上升到最近公共祖先再下降

2如果st是end的父节点 那么st就是lca 

3如果st是end的子节点 那么end就是lca

所以必定经过lca

再思考 从st到lca到end 是怎样的路程  首先不断上升 再下降

上升可以反过来想 从lca下降到st 再把路程中的L R换成U

所以就需要dfs 求出lca到st 和end的路程

```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    TreeNode* lca(TreeNode* root, int startValue, int destValue)
    {
        if (root == nullptr || root->val == startValue || root->val == destValue)
        return root;

        TreeNode *left = lca(root->left, startValue, destValue);
        TreeNode *right = lca(root->right, startValue, destValue);

        if (left && right) return root;
        return left ? left : right;
    }
    string res;
    void dfs(TreeNode* cur, int t, string& path)
    {
        if (cur == nullptr)
        return ;
        if (cur->val == t)
        {
            res = path;
            return;
        }
        if (cur->left)
        {
            path += 'L';
            dfs(cur->left, t, path);
            path.pop_back();
        }
        if (cur->right)
        {
            path += 'R';
            dfs(cur->right, t, path);
            path.pop_back();
        }
        return ;
    }
    string getDirections(TreeNode* root, int startValue, int destValue) 
    {
        TreeNode* newroot = lca(root, startValue, destValue);

        string path = "";
        res = "";
        dfs(newroot, startValue, path);
        string ans(res.size(), 'U');
        res = "";
        dfs(newroot, destValue, path);
        ans += res;
        return ans;
    }
};
```

