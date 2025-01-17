### 25.K个一组翻转链表
**题目大意：**
```angular2html
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。
k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。
你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。
```
**解题思路：**
```angular2html
翻转链表只需要翻转每一个节点的指针即可。

1.写一个辅助函数————翻转一个子链表，并返回新的头与尾
注意：需要初始化prev为tail的下一个节点，表示反转后链表的尾部。（因为后面主函数处理子链表时，子链表反转后尾部要接翻转前的尾部）

2.写主函数————翻转链表（K个一组）
注意：创建一个虚拟头节点hair，hair.next = head

```
**复杂度分析：**
```angular2html
时间复杂度：O(n)，其中 n 为链表的长度。head 指针会在 O(n/k) 个节点上停留，每次停留需要进行一次 O(k) 的翻转操作。
空间复杂度：O(1)，我们只需要建立常数个变量。
```
**完整代码：**
```angular2html
class ListNode:
    def __init__(self, val = 0, next = None):
        self.val = val
        self.next = next

# 辅助函数：翻转一个子链表，并返回新的头与尾
def reverse(self, head: ListNode, tail: ListNode):
    # 初始化prev为tail的下一个节点，表示反转后链表的尾部
    prev = tail.next
    p = head  # p指向子链表的头节点
    # 反转链表，直到p到达tail节点
    while prev != tail:
        nex = p.next  # 保存p的下一个节点
        p.next = prev  # 反转p的指针，指向prev
        prev = p  # prev向前移动
        p = nex  # p向前移动
    return tail, head  # 返回反转后的新的头节点（tail）和尾节点（head）

# 主函数
def reverseKGroup(self, head: ListNode, k: int) -> ListNode:
    # 创建一个虚拟头节点hair，方便处理头节点的反转
    hair = ListNode(0)
    hair.next = head  # hair指向原链表的头
    pre = hair  # pre指向当前处理的子链表的前一个节点
    
    # 开始循环，直到head为None（即遍历完整个链表）
    while head:
        tail = pre  # tail指针指向head的前一个节点pre
        
        # 检查剩余部分的节点数是否大于等于k
        for i in range(k):
            tail = tail.next  # 移动tail指针，向后找到第k个节点
            if not tail:  # 如果剩余的节点少于k个，直接返回已处理的链表
                return hair.next
        # 如果剩余的节点大于等于k，那么tail指向从head开始的第k个节点
        
        # 保存tail之后的节点，供后续连接使用
        nex = tail.next
        # 反转从head到tail的子链表
        head, tail = reverse(head, tail)
        
        # 把反转后的子链表重新接回原链表(注意，这里很关键)
        pre.next = head  # pre指向反转后的子链表头
        tail.next = nex  # 反转后的子链表尾连接到nex（即原链表中tail的下一个部分）
        
        # 更新pre和head，继续处理下一个子链表
        pre = tail  # pre更新为反转后子链表的尾节点
        head = tail.next  # head更新为反转后子链表尾节点的下一个节点
    
    return hair.next  # 返回反转后的链表，hair.next指向链表的头节点
```    

### 138.随机链表的复制
**题目大意：**
```angular2html
给你一个长度为 n 的链表，每个节点包含一个额外增加的随机指针 random ，该指针可以指向链表中的任何节点或空节点。
构造这个链表的 深拷贝。 深拷贝应该正好由 n 个 全新 节点组成，其中每个新节点的值都设为其对应的原节点的值。新节点的 next 指针和 random 指针也都应指向复制链表中的新节点，并使原链表和复制链表中的这些指针能够表示相同的链表状态。复制链表中的指针都不应指向原链表中的节点。
例如，如果原链表中有 X 和 Y 两个节点，其中 X.random --> Y 。那么在复制链表中对应的两个节点 x 和 y ，同样有 x.random --> y 。
返回复制链表的头节点。
用一个由 n 个节点组成的链表来表示输入/输出中的链表。每个节点用一个 [val, random_index] 表示：
val：一个表示 Node.val 的整数。
random_index：随机指针指向的节点索引（范围从 0 到 n-1）；如果不指向任何节点，则为  null 。
你的代码只接受原链表的头节点 head 作为传入参数。
```
**解题思路：**
```angular2html
回溯 & 哈希表
原链表的节点和新链表的节点交替出现

Steps:
1.复制节点：在遍历链表的过程中，我们将每个节点的复制节点插入到它的后面。（最终实现原链表的节点和新链表的节点交替出现）
2.处理random指针：新节点的 random 应该指向原链表中 random 指针所指向的下一个节点。（因为复制的节点在原节点的下一个）
3.拆分链表：拆分成两个独立的链表————一个原链表，一个新链表

```
**复杂度分析：**
```angular2html
时间复杂度：O(n),我们只遍历了链表三次：一次是复制节点，一次是复制 random 指针，最后一次是拆分链表。
空间复杂度：我们只用了 O(1) 的额外空间（除了返回的新链表）
```
**完整代码：**
```angular2html
class Node:
    def __init__(self, x:int, next = None, random = None):
        self.val = x
        self.next = next
        self.random = random

class Solution:
    def copyRandomList(self, head:Node) -> Node:
        if not head:
            return None
    
        # 1.复制节点，并将复制节点插入原链表中
        current = head
        while current:
            new_node = Node(current.val)  # 创建一个新节点
            new_node.next = current.next  # 新节点的 next 指向原链表的下一个节点
            current.next = new_node  # 原节点的 next 指向新节点
            current = new_node.next  # 移动到下一个原节点
            
        # 2.复制 random 指针（新节点的 random 应该指向原链表中 random 指针所指向的下一个节点。）
        current = head
        while current:
            if current.random:
                current.next.random = current.random.next  # 复制 random 指针
            current = current.next.next  # 跳过新节点，回到下一个原节点

        # 3.拆分链表，恢复原链表并提取出新链表
        current = head
        new_head = head.next  # 新链表的头节点
        while current:
            new_node = current.next
            current.next = new_node.next  # 恢复原链表的next指针
            if new_node.next:
                new_node.next = new_node.next.next  # 将新链表节点的 next 指向下一个新节点
            current = current.next  # 移动到下一个原节点
            
        return new_head
```

### 148.排序链表
**题目大意：**
```angular2html
给你链表的头结点 head ，请将其按 升序 排列并返回 排序后的链表。
```
**解题思路：**
```angular2html
归并排序————将链表分成两部分，分别排序后再合并起来。由于链表支持顺序访问和分割，因此归并排序可以非常高效地应用于链表。
Steps:
1.分割链表：通过快慢指针找到链表的中间节点，将链表分成两部分。（原理：快指针 fast 每次走两步，慢指针 slow 每次走一步。当快指针走到链表末尾时，慢指针恰好指向中间节点。）
2.递归排序：递归地对这两部分链表进行排序。
3.合并两个排序好的链表：将两个排序好的子链表合并成一个有序的链表。
```
**复杂度分析：**
```angular2html
时间复杂度：每次分割链表都会将链表拆成两半，因此递归的深度是 O(log n)，每次合并的时间复杂度是 O(n)，所以总的时间复杂度为 O(n log n)
空间复杂度：归并排序的递归调用栈的空间复杂度是 O(log n)，而每次合并链表的操作只需要 O(1) 的额外空间。所以，空间复杂度为 O(log n)
```
**完整代码：**
```angular2html
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def sortList(self, head: ListNode) -> ListNode:
        # 递归排序函数，head为当前链表的头节点，tail为分割的边界节点（为空表示链表的尾部）
        def sortFunc(head: ListNode, tail: ListNode) -> ListNode:
            # 基本情况：如果链表为空，直接返回空链表
            if not head:
                return head
            # 基本情况：如果链表只剩下一个节点，直接返回该节点
            # head.next == tail意味着只有一个节点
            if head.next == tail:
                head.next = None  # 注意：如果直接返回 head 而不设置 head.next = None，递归的下一层调用中可能会导致链表结构仍然连接在一起（归并排序中，链表分割时必须保证左右两部分是完全独立的链表，避免分割错误。）
                return head

            # 使用快慢指针找到链表的中间节点
            slow = fast = head
            # 快慢指针遍历链表，找到中点
            # fast指针每次走两步，slow指针每次走一步
            # 当fast指针到达尾部时，slow指针就指向链表的中间节点
            while fast != tail:
                slow = slow.next
                fast = fast.next
                if fast != tail:
                    fast = fast.next

            # 中间节点
            mid = slow
            # 分别递归地对左半部分和右半部分链表进行排序，并合并
            return merge(sortFunc(head, mid), sortFunc(mid, tail))

        # 合并两个排序好的链表，返回合并后的有序链表
        def merge(head1: ListNode, head2: ListNode) -> ListNode:
            # 创建一个虚拟头节点dummyHead，用于简化操作
            dummyHead = ListNode(0)
            temp, temp1, temp2 = dummyHead, head1, head2

            # 遍历两个链表，按顺序选择较小的节点连接到合并链表中
            while temp1 and temp2:
                if temp1.val <= temp2.val:
                    temp.next = temp1
                    temp1 = temp1.next
                else:
                    temp.next = temp2
                    temp2 = temp2.next
                temp = temp.next

            # 如果有剩余的节点，直接连接到合并链表的末尾
            if temp1:
                temp.next = temp1
            elif temp2:
                temp.next = temp2
            
            # 返回合并后的链表，从dummyHead.next开始
            return dummyHead.next
        
        # 从头开始对整个链表进行排序
        return sortFunc(head, None)
``` 

### 23.合并k个升序链表
**题目大意：**
```angular2html
给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。
```
**解题思路：**
```angular2html
使用优先队列（小顶堆）

Steps:
使用一个虚拟头结点来构建新链表。
1.初始化一个小顶堆，并将每个链表的头结点加入堆中。
2.每次从堆中取出最小的节点，将它接到新链表的尾部，接着将该节点的下一个节点加入堆中。
3.重复1-2过程直到所有链表都被合并完。
```
**复杂度分析：**
```angular2html
时间复杂度：每次从堆中弹出一个节点并插入一个节点的操作时间复杂度是O(logk)，其中 k 是链表的数量。总共有 n 个节点，因此总的时间复杂度为 O(nlogk)
空间复杂度：使用了一个大小为 k 的堆存储每个链表的头节点，因此空间复杂度是O(k)
```
**完整代码：**
```
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        # 创建一个小顶堆
        heap = []
        
        # 将每个链表的头结点放入堆中
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(heap, (lists[i].val, i, lists[i]))  # 堆中存储元组 (值, 链表索引, 节点)
        
        # 创建一个虚拟头节点
        dummy = ListNode()
        current = dummy
        
        # 逐一从堆中弹出最小节点并将其接到合并链表中
        while heap:
            val, index, node = heapq.heappop(heap)
            current.next = node
            current = node
            
            # 如果弹出的节点有下一个节点，将下一个节点放入堆中
            if node.next:
                heapq.heappush(heap, (node.next.val, index, node.next))
                
        return dummy.next
``` 

### 146.LRU缓存
**题目大意：**
```angular2html
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。
```
**解题思路：**
```angular2html
哈希表+双向链表
双向链表：靠近头部的键值对是最近使用的，而靠近尾部的键值对是最久未使用的。
哈希表：通过缓存 数据的键 映射到其在 双向链表中的位置。

对于get操作：
    如果 key 不存在，则返回 −1；
    如果 key 存在，则 key 对应的节点将变更为最近被使用的节点（因为当前get它了）。通过哈希表定位到该节点在双向链表中的位置，并将其移动到双向链表的头部，最后返回该节点的值。
对于put操作：
    如果 key 不存在，使用 key 和 value 创建一个新的节点，在双向链表的头部添加该节点，并将 key 和该节点添加进哈希表中。然后判断双向链表的节点数是否超出容量，如果超出容量，则删除双向链表的尾部节点，并删除哈希表中对应的项；
    如果 key 存在，则与 get 操作类似，先通过哈希表定位，再将对应的节点的值更新为 value，并将该节点移到双向链表的头部。

此外，在双向链表的实现中，使用一个伪头部（dummy head）和伪尾部（dummy tail）标记界限。
```
**复杂度分析：**
```angular2html
时间复杂度：对于 put 和 get 都是 O(1)。
空间复杂度：O(capacity)，因为哈希表和双向链表最多存储 capacity+1 个元素
```
**完整代码：**
```
class DLinkedNode:
    """定义双向链表"""
    def __init__(self, key=0, value=0):
        # 每个节点存储一个键值对
        self.key = key
        self.value = value
        # 指向先后节点指针
        self.prev = None
        self.next = None

class LRUCache:

    def __init__(self, capacity:int):
        # 初始化缓存
        self.cache = dict()  # 哈希表，存储 key 到节点的映射
        
        # 使用伪头部和伪尾部节点
        self.head = DLinkedNode()
        self.tail = DLinkedNode()
        self.head.next = self.tail
        self.tail.prev = self.head
        
        self.capacity = capacity
        self.size = 0
        
    def get(self, key:int) -> int:
        """获取缓存中指定key的值"""
        
        if key not in self.cache:
            return -1  # 如果 key 不在缓存中，返回 -1
        # 如果 key 存在，先通过哈希表定位节点，再将节点移到头部
        node = self.cache[key]
        self.moveToHead(node)  # 将该节点移动到头部，表示最近使用
        return node.value
        
    def put(self, key:int, value:int) -> None:
        """插入或更新缓存中的 key-value 对"""
       
        # 如果 key 不在缓存中，创建一个新的节点
        if key not in self.cache:
            node = DLinkedNode(key, value)
            self.cache[key] = node  # 将节点加入哈希表
            self.addToHead(node)  # 将节点添加到链表头部
            self.size += 1
            
            # 如果缓存超出容量，移除最少使用的节点（链表尾部节点）
            if self.size > self.capacity:
                removed = self.removeTail()  # 移除尾部节点
                self.cache.pop(removed.key)  # 从哈希表中删除该节点
                self.size -= 1
        else:
            # 如果 key 已存在，更新其值并将节点移动到头部
            node = self.cache[key]
            node.value = value
            self.moveToHead(node)
            
    def addToHead(self, node):
        """将节点添加到链表头部"""
        node.prev = self.head  # 当前节点的 prev 指向伪头节点
        node.next = self.head.next  # 当前节点的 next 指向原来的头部节点
        self.head.next.prev = node  # 原头部节点的 prev 指向当前节点
        self.head.next = node  # 伪头节点的 next 指向当前节点
        
    def removeNode(self, node):
        # 从链表中移除节点
        node.prev.next = node.next  # 前一个节点的 next 指向当前节点的 next
        node.next.prev = node.prev  # 后一个节点的 prev 指向当前节点的 prev
        
    def moveToHead(self, node):
        # 将一个节点移到链表头部
        self.removeNode(node)  # 先从链表中移除该节点
        self.addToHead(node)  # 再将其添加到链表头部
        
    def removeTail(self):
        # 移除链表尾部的节点（即最久未使用的节点）
        node = self.tail.prev  # 获取尾部节点
        self.removeNode(node)  # 从链表中移除该节点
        return node  # 返回被移除的节点
``` 





