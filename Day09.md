### 102.二叉树的层序遍历
**题目大意：**
```angular2html
给你二叉树的根节点 root ，返回其节点值的 层序遍历 。 （即逐层地，从左到右访问所有节点）。
```
**解题思路：**
```angular2html
队列
```
**复杂度分析：**
```angular2html
时间复杂度：每个点进队出队各一次，故渐进时间复杂度为 O(n)。
空间复杂度：队列中元素的个数不超过 n 个，故渐进空间复杂度为 O(n)。
```
**完整代码：**
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def levelOrder(self, root:TreeNode) -> List[List[int]]:
        res = []
        if not root:
            return res
         
        queue = deque([root])  # 初始化队列，并加入根节点
        
        # 依次遍历各层
        while queue:
            level = []  # 存储当前层的遍历值
            size = len(queue)  # 当前层的节点数
            # 遍历当前层
            for _ in range(size):
                node = queue.popleft()  # 队列出队（获取当前节点）
                level.append(node.val)  # 保存节点值
                # 如果当前节点存在 左子节点
                if node.left:
                    queue.append(node.left)  # 左子节点入队(解释一下：下一轮实现从左往右遍历 左子节点、右子节点)
                # 如果当前节点存在 右子节点
                if node.right is not None:
                    queue.append(node.right)  # 右子节点入队
            res.append(level)  # 将当前层结果列表存入res中
        return res
```    

### 108. 将有序数组转换为二叉搜索树
**题目大意：**
```angular2html
给你一个整数数组 nums ，其中元素已经按 升序 排列，请你将其转换为一棵平衡二叉搜索树。
```
**解题思路：**
```angular2html
二叉搜索树（BST）：对任意节点，左子树的值 < 节点的值 < 右子树的值。
平衡树：每个节点的左右子树高度之差（平衡因子）不超过 1

因为数组已经是升序排列的，所以直接利用中间元素来构造平衡二叉树的根节点
    1. 选择中间元素作为根节点。
    2. 将数组的左半部分递归构建为左子树，右半部分递归构建为右子树。
递归构建过程会自然地保证树的平衡，因为每次递归都选择数组的中间元素。
```
**复杂度分析：**
```angular2html
时间复杂度：O(n)，其中 n 是数组的长度。每次递归操作都对数组中的一个元素进行处理，因此总的时间复杂度是 O(n)。
空间复杂度：O(n)，递归的最大深度为 O(log n)，但是由于树的构建还需要存储节点，所以总的空间复杂度是 O(n)。
```
**完整代码：**
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def sortedArrayToBST(self, nums:List[int]) -> TreeNode:
        # 辅助函数：递归构建平衡二叉树
        def helper(left, right):
            # 递归终止条件：当左指针超过右指针时，返回 None (注：当左右指针相等时，就会构建当前指针对应元素的节点)
            if left > right:
                return None
            
            # 选择中间元素作为根节点
            mid = (left + right) // 2
            root = TreeNode(nums[mid])
            
            # 递归构建左右子树
            root.left = helper(left, mid-1)
            root.right = helper(mid+1, right)
            
            return root
            
        # 调用辅助函数，开始递归构建树
        return helper(0, len(nums)-1)
            
```

### 98.验证二叉搜索树
**题目大意：**
```angular2html
给你一个二叉树的根节点 root ，判断其是否是一个有效的二叉搜索树。
```
**解题思路：**
```angular2html
递归
1.对于每个节点，检查其值是否在一个有效的区间内：
    对于根节点，它的值没有任何限制。
    对于左子树节点，它的值必须小于父节点的值。
    对于右子树节点，它的值必须大于父节点的值。
2.递归时，传递一个合法的值范围，确保每个节点的值满足该范围。
3.如果某个节点的值不满足合法的范围，则该树不是一个有效的二叉搜索树。
```
**复杂度分析：**
```angular2html
时间复杂度：O(n)，其中 n 是二叉树的节点数。我们需要遍历每个节点一次来检查其合法性，因此时间复杂度是 O(n)。
空间复杂度：O(h)，其中 h 是树的高度。由于递归调用的栈深度最多为树的高度，空间复杂度为 O(h)。在最坏情况下，树是链状的，空间复杂度为 O(n)；在平衡的树中，空间复杂度为 O(log n)。
```
**完整代码：**
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def isValidBST(self, root:TreeNode) -> bool:
        # 辅助函数，判断二叉树是否满足 BST 的要求
        def helper(node, lower, upper):
            # 递归结束，返回True
            if not node:
                return True
            
            val = node.val
            # 当前节点值必须在 (lower, upper) 范围内，否则不满足条件
            if val <= lower or val >= upper:
                return False
                
            # 递归检查右子树，当前节点值是下限，递归检查左子树，当前节点值是上限
            return helper(node.right, val, upper) and helper(node.left, lower, val)
            
        # 从根节点开始递归
        return helper(root, float('-inf'), float('inf'))
``` 

### 230.二叉搜索树中第 K 小的元素
**题目大意：**
```angular2html
给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 小的元素（从 1 开始计数）。
```
**解题思路：**
```angular2html
中序遍历二叉搜索树会得到一个升序排列序列。
因此，通过中序遍历，可以在遍历时按顺序找到第k小的元素。
Steps：
    1. 中序遍历过程中，保持一个计数器，记录已访问的节点数量，直至计数器达到k
    2. 可以用递归的方式实现中序遍历
```
**复杂度分析：**
```angular2html
时间复杂度：O(n)，其中 n 是树中的节点数。因为我们需要对树进行中序遍历，遍历每个节点一次。
空间复杂度：O(h)，其中 h 是树的高度。递归调用栈的空间复杂度取决于树的高度。在最坏情况下，树是链状的，空间复杂度为 O(n)；在平衡树中，空间复杂度为 O(log n)。
```
**完整代码：**
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def kthSmallest(self, root:TreeNode, k:int) -> int:
        self.count = 0  # 计数器：记录遍历节点数
        self.res = 0  # 记录遍历节点
        
        # 辅助函数：中序遍历
        def in_order(node):
            # 递归结束条件
            if not node:
                return 
            
            # 遍历左子树
            in_order(node.left)
            
            # 访问当前节点
            self.count += 1  # 计数器+1
            if self.count == k:
                self.res = node.val
                return 
            
            # 遍历右子树 
            in_order(node.right)
            
        # 从根节点开始遍历
        in_order(root)
        return self.res    
``` 

### 199.二叉树的右视图
**题目大意：**
```angular2html
给定一个二叉树的 根节点 root，想象自己站在它的右侧，按照从顶部到底部的顺序，返回从右侧所能看到的节点值。
```
**解题思路：**
```angular2html
从二叉树的右侧看，我们会看到每一层的最右边的节点。因此，我们可以通过层序遍历（BFS）来逐层遍历树，并在每一层的遍历中记录最右侧的节点。
Steps:
    1. 层序遍历：通过队列实现层序遍历，逐层访问节点。
    2. 记录每一层的最右节点：对于每一层，取该层的最后一个节点的值，因为它就是右侧最能看到的节点。
    3. 返回结果：将每一层的最右节点的值按顺序返回。
```
**复杂度分析：**
```angular2html
时间复杂度：O(n)，其中 n 是二叉树中的节点数。我们需要访问每个节点一次。
空间复杂度：O(n)，最坏情况下（即树的宽度最大时），队列需要存储所有的节点。队列的最大长度为树的最大宽度，最坏情况是完全二叉树时，空间复杂度为 O(n)
```
**完整代码：**
```
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def rightSideView(self, root:TreeNode) -> List[int]:
        if not root:
            return []
        
        res = []  # 记录结果
        queue = deque([root])  # 初始化队列
        
        while queue:
            level_length = len(queue)  # 当前层的节点数量
            for i in range(level_length):
                node = queue.popleft()  # 取出队列中的节点
                # 如果是当前层的最后一个节点，则添加到结果中
                if i == level_length - 1:
                    res.append(node.val)
                # 将左右节点加入队列
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)    
                    
        return res
```





