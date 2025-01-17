### 94.二叉树的中序遍历
**题目大意：**
```angular2html
给定一个二叉树的根节点 root ，返回 它的 中序 遍历 。
```
**解题思路：**
```angular2html
深度优先遍历
递归
```
**复杂度分析：**
```angular2html
时间复杂度：O(n)，其中 n 为二叉树节点的个数。二叉树的遍历中每个节点会被访问一次且只会被访问一次。
空间复杂度：O(n)。空间复杂度取决于递归的栈深度，而栈深度在二叉树为一条链的情况下会达到 O(n) 的级别。
```
**完整代码：**
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def inorderTraversal(self, root:TreeNode) -> List[int]:
        res = []  # 存储遍历结果
        
        def in_order(node):
            if not node:
                return
                
            # 访问优先级：左子树 -> 根节点 -> 右子树
            in_order(node = node.left)
            res.append(node.val)
            in_order(node = node.right)
        
        in_order(root)
        return res
        

```    

### 104.二叉树的最大深度
**题目大意：**
```angular2html
给定一个二叉树 root ，返回其最大深度。

二叉树的 最大深度 是指从根节点到最远叶子节点的最长路径上的节点数。
```
**解题思路：**
```angular2html
深度优先搜索
递归

若左子树和右子树的最大深度 l 和 r，那么该二叉树的最大深度为:
max(l,r)+1
```
**复杂度分析：**
```angular2html
时间复杂度：O(n)，其中 n 为二叉树节点的个数。每个节点在递归中只被遍历一次。
空间复杂度：O(height)，其中 height 表示二叉树的高度。递归函数需要栈空间，而栈空间取决于递归的深度，因此空间复杂度等价于二叉树的高度。
```
**完整代码：**
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxDepth(self, root:TreeNode) -> int:
        # 如果当前节点为空，深度为 0
        if root is None: 
            return 0
        else:
            # 递归求左子树的最大深度
            left_height = self.maxDepth(root.left)
            # 递归求右子树的最大深度
            right_height = self.maxDepth(root.right) 
            # 当前节点的最大深度是左子树和右子树最大深度中的较大值 + 1（包含当前节点）
            return max(left_height, right_height) + 1
``` 

### 226.翻转二叉树
**题目大意：**
```angular2html
给你一棵二叉树的根节点 root ，翻转这棵二叉树，并返回其根节点。
```
**解题思路：**
```angular2html
递归
将每一个节点的左右子树交换，从而得到其镜像：
    从根节点开始，递归地对树进行遍历，并从叶子节点先开始翻转。
    如果当前遍历到的节点 root 的左右两棵子树都已经翻转，那么我们只需要交换两棵子树的位置
```
**复杂度分析：**
```angular2html
时间复杂度：O(N)，其中 N 为二叉树节点的数目。我们会遍历二叉树中的每一个节点，对每个节点而言，我们在常数时间内交换其两棵子树。
空间复杂度：O(N)。使用的空间由递归栈的深度决定，它等于当前节点在二叉树中的高度。在平均情况下，二叉树的高度与节点个数为对数关系，即 O(logN)。而在最坏情况下，树形成链状，空间复杂度为 O(N)。
```
**完整代码：**
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def invertTree(self, root:TreeNode) -> TreeNode:
        if not root:
            return root
        
        # 递归反转左子树和右子树
        left = self.invertTree(root.left)
        right = self.invertTree(root.right)
        # 交换当前节点的左右子树
        root.left, root.right = right, left
        
        return root
``` 

### 101.对称二叉树
**题目大意：**
```angular2html
给你一个二叉树的根节点 root ， 检查它是否轴对称。
```
**解题思路：**
```angular2html
递归地比较树的左右子树:
    树的左子树和右子树的根节点应该具有相同的值。
    左子树的左子树应该与右子树的右子树对称。
    左子树的右子树应该与右子树的左子树对称。
```
**复杂度分析：**
```angular2html
时间复杂度：O(n)。其中 n 是二叉树的节点数。我们需要遍历每个节点一次来判断是否对称。
空间复杂度：O(n)。由于递归调用栈的深度与树的高度h成正比，因此空间复杂度为 O(h)。最坏情况下，树是单链状的，h 为 n
```
**完整代码：**
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSymmetric(self, root:TreeNode) -> bool:
        # 辅助函数，判断两棵树是否是镜像对称的
        def isMirror(t1:TreeNode, t2:TreeNode) -> bool:
            # 如果两个节点都为空，返回true
            if not t1 and not t2:
                return True
            # 如果一个为空，一个不为空，返回False
            if not t1 or not t2:
                return False
            # 判断当前节点值是否相等，并递归判断左子树和右子树是否对称
            return (t1.val == t2.val) and isMirror(t1.left, t2.right) and isMirror(t1.right, t2.left)
            
        # 从根节点开始判断
        if not root:
            return True  # 空树是对称的
        
        # 判断左子树和右子树是否是镜像对称的
        return isMirror(root.left, root.right)
``` 

### 543.二叉树的直径
**题目大意：**
```angular2html
给你一棵二叉树的根节点，返回该树的 直径 。

二叉树的 直径 是指树中任意两个节点之间最长路径的 长度 。这条路径可能经过也可能不经过根节点 root 。

两节点之间路径的 长度 由它们之间边数表示。
```
**解题思路：**
```angular2html
递归

对于任意节点 node，直径由以下两部分组成：
    左子树的高度：从节点 node 到左子树最深的叶子节点的路径长度。
    右子树的高度：从节点 node 到右子树最深的叶子节点的路径长度。
```
**复杂度分析：**
```angular2html
时间复杂度：O(n)，其中 n 是二叉树的节点数。每个节点都被访问一次，因此时间复杂度是 O(n)。
空间复杂度：O(Height)，其中 Height 为二叉树的高度。
```
**完整代码：**
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def diameterOfBinaryTree(self, root:TreeNode) -> int:
        self.ans = 0  # 记录树的直径
        
        # 定义递归函数，返回树的深度
        def depth(node):
            # 访问到空节点了，返回0
            if not node:
                return 0
                
            # 递归计算左子树和右子树的深度
            L = depth(node.left)
            R = depth(node.right)
            
            # 计算d_node即L+R+1 并更新ans
            self.ans = max(self.ans, L + R)
            # 返回该节点为根的子树的深度
            return max(L, R)+1
        # 计算二叉树的深度，同时更新树的直径
        depth(root)
        # 返回最终的直径
        return self.ans
``` 





