from collections import deque


class Node:
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __str__(self):
        return str(self.data)


class BST:  # 用于动态查找·删除·增加序列,度为0的个数=度为2的个数+1
    def __init__(self, root=None, key=lambda x, y: x < y):  # 默认升序
        self.__root = root
        self.key = key

    @property
    def root(self):
        return self.__root

    def find(self, element):  # 时间复杂度是O(logn)
        self._hot = None  # 指向待查节点父节点
        self._pointer = self.__root  # 指向待查节点
        while self._pointer:
            if self._pointer.data == element:  # 注意顺序,应为self.key规则可能包含=
                return True
            self._hot = self._pointer
            if self.key(element, self._pointer.data):
                self._pointer = self._pointer.left
            else:
                self._pointer = self._pointer.right
        return False

    def add(self, element):  # 树的高度越小,效率越高,这就要求插入的序列尽量无序,一定是在叶子节点处增加新节点
        if self.find(element):  # 只插入不重复的序列
            return False
        node = Node(element)
        if self._hot:
            if self.key(element, self._hot.data):
                self._hot.left = node
            else:
                self._hot.right = node
        else:  # 空树
            self.__root = node
        return True

    def delete(self, element):  # 时间复杂度是O(logn)
        """
        1. 删除的节点只有左/右孩子,直接替换掉左/右孩子
        2. 删除的节点有左右孩子,先找到右孩子,如果他还有左孩子,就一直迭代下去,然后拷贝该孩子的值替换掉待删结点(也可以找左孩子的最大值结点)
        """
        if self.find(element):
            if self._pointer.left and self._pointer.right:
                first = self._pointer
                second = self._pointer.right
                while second.left:
                    first = second
                    second = second.left
                self._pointer.data = second.data
                if first is self._pointer:  # 注意此处要分两种情况
                    first.right = second.right
                else:
                    first.left = second.right

            else:
                if self._pointer.left:  # 右子树空
                    son = self._pointer.left
                else:  # 左子树空
                    son = self._pointer.right
                if not self._hot:  # 注意此种情况
                    self.__root = son
                elif self.key(self._hot.data, self._pointer.data):
                    self._hot.right = son
                else:
                    self._hot.left = son
            return True
        return False

    def post_order_copy(self):  # 思想类似于归并
        def _post_order_copy(node):
            if node:
                left = _post_order_copy(node.left)
                right = _post_order_copy(node.right)
                return Node(node.data, left, right)

        return _post_order_copy(self.__root)

    def is_same(self, another):
        def _is_same(first, second):
            if first and second:
                return (first.data == second.data) and _is_same(first.left, second.left) and _is_same(first.right,
                                                                                                      second.right)
            else:
                return not (first or second)

        return _is_same(self.__root, another.__root)

    def pre_order(self):
        def _pre_order(root):
            print(root, end='\t')
            # root.left,root.right=root.right,root.left #反转二叉树
            if root.left:  # 减小递归深度
                _pre_order(root.left)
            if root.right:
                _pre_order(root.right)

        if self.__root:
            _pre_order(self.__root)

    def pre_order_stack(self):
        stack = []
        root = self.__root
        while root or stack:
            if root:
                print(root)
                if root.right:  # 提高效率
                    stack.append(root.right)  # 也可以让root入栈,出栈时令root=stack.pop().right
                root = root.left
                # root.left,root.right=root.right,root.left  # 反转二叉树
                # root=root.right 
            else:
                root = stack.pop()

        # 频繁出入栈，效率低
        # stack=[self.__root]
        # while stack:
        #     root=stack.pop()
        #     if root:
        #         print(root)
        #         stack.append(root.right)
        #         stack.append(root.left)

    def in_order(self):
        def _in_order(root):
            if root.left:
                _in_order(root.left)
            print(root)
            if root.right:
                _in_order(root.right)

        if self.__root:
            _in_order(self.__root)

    def in_order_stack(self):  # 稳定排序,思考如何按从大到小排序
        stack = []
        root = self.__root
        while root or stack:
            if root:
                stack.append(root)
                root = root.left
            else:
                root = stack.pop()
                print(root)
                root = root.right

    def post_order(self):
        def _post_order(root):
            if root.left:
                _post_order(root.left)
            if root.right:
                _post_order(root.right)
            print(root)

        if self.__root:
            _post_order(self.__root)

    def post_order_stack(self):  # 后续遍历二叉树还可以用于打印从根节点到叶节点的路径
        # method 1:  推荐
        root = self.__root
        stack = []  # 栈中的结点已经不包含左节点
        r_child = None  # 记录最后一次打印的结点(如果结点有重复值,此方法会失效)
        while root:  # 这一步必须单独入栈,没法跟下面的入栈代码合并
            stack.append(root)
            root = root.left
        while stack:
            root = stack[-1].right
            if root and root != r_child:
                while root:
                    stack.append(root)
                    root = root.left
            else:
                r_child = stack.pop()
                print(r_child)
                # if not(r_child.left or r_child.right):  # 打印root -> leaf路径
                #     for i in stack:
                #         print(i,end=' ')
                #     print(r_child,end='\n')

        # method 2:  可以看成DRL,然后再将结果翻转
        # stack=[]
        # result=[]
        # root=self.__root        
        # while root or stack:
        #     if root:
        #         result.append(root.data)
        #         if root.left:
        #             stack.append(root.left)
        #         root=root.right  
        #     else:
        #         root=stack.pop()
        # while result:
        #     print(result.pop())

        # method 3: 不推荐
        # root=self.__root
        # stack=[self.__root]
        # while root or stack:
        #     if root:
        #         if stack and stack[-1]==root:
        #             if root.right:
        #                 stack.append(root.right)
        #                 stack.append(root.right)
        #             if root.left:
        #                 stack.append(root.left)
        #             root=root.left
        #         else:
        #             print(root)
        #             root=None
        #     else:
        #         root=stack.pop()

    def BFS(self):
        count = 0  # 结点个数
        if self.__root:
            queue = deque([self.__root])
            while queue:
                node = queue.popleft()
                count += 1
                print(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return count

    def find_path(self):  # root => leaf路径问题, 此处是先序遍历
        path = []

        def _find_path(root):
            path.append(root.data)
            if root.left:
                _find_path(root.left)
            if root.right:
                _find_path(root.right)
            if root.left is None and root.right is None:
                print(path)
            path.pop()

        if self.__root:
            _find_path(self.__root)

    def find_path_stack(self):  # all path from root to leaf
        root = self.__root
        stack = []
        path = []
        while root or stack:
            if root:
                path.append(root.data)
                if root.right:
                    stack.append(root)  # 这里不能让其右孩子入栈
                elif not root.left:
                    print(path)
                root = root.left
            else:
                root = stack.pop()
                if root.left:  # 一定要判断
                    node = path.pop()
                    l_data = root.left.data
                    while l_data != node:
                        node = path.pop()
                root = root.right

    def find_level(self, element):  # 可以求二叉树最大宽度
        # level = 0
        # if self.__root:
        #     queue = deque((level+1, self.__root))
        #     while queue:
        #         node_or_level = queue.popleft()
        #         if isinstance(node_or_level, int):
        #             if queue:
        #                 level = node_or_level
        #                 queue.append(node_or_level+1)
        #             else:
        #                 level = 0
        #         else:
        #             if node_or_level.data == element:
        #                 break
        #             if node_or_level.left:
        #                 queue.append(node_or_level.left)
        #             if node_or_level.right:
        #                 queue.append(node_or_level.right)
        # return level

        # 低效版
        depth = -1

        def _find(root, level):
            nonlocal depth
            if root.data == element:
                depth = level
            else:
                if root.left:
                    _find(root.left, level + 1)
                if root.right:
                    _find(root.right, level + 1)

        if self.__root:
            _find(self.__root, 1)
        return depth

    def node_num(self, node):  # 各种遍历都可以计算结点个数
        if node:
            return self.node_num(node=node.left) + self.node_num(node=node.right) + 1
        else:
            return 0

    def max_depth(self):  # BFS也可以计算最大深度
        def _max_depth(node):
            if node:
                return max(_max_depth(node.left), _max_depth(node.right)) + 1
            else:
                return 0

        return _max_depth(self.__root)

    def find_common_parent(self, one, another):  # one,another不一定存在,如果最低公共父节点不存在,返回None
        def _find(root, _one, _another):
            if root:
                if root.data == _one or root.data == _another:
                    if (root.data == _one) and self.find(_another):
                        return root.data
                    if (root.data == _another) and self.find(_one):
                        return root.data
                else:
                    if (_one < root.data < _another) or (_another < root.data < _one):
                        if self.find(_one) and self.find(_another):
                            return root.data
                    elif (_one < root.data) and (_another < root.data):
                        return _find(root.left, _one, _another)
                    else:
                        return _find(root.right, _one, _another)

        return _find(self.__root, one, another)

    def find_common_parent_stack(self, child, sibling):  # 寻找最低公共父节点
        root = self.__root
        stack = []
        path = []
        while root or stack:
            if root:
                if root.data == child:  # 找到了child,则查看child的所有父节点中哪个也属于sibling父节点即可
                    return __class__.check(path, sibling)
                path.append(root)
                if root.right:
                    stack.append(root)
                root = root.left
            else:
                root = stack.pop()
                if root.left:
                    node = path.pop()
                    while root.left != node:
                        node = path.pop()
                root = root.right

    def check(path, sibling):
        queue = deque()
        while path:
            parent = path.pop()
            if parent.left:
                queue.append(parent.left)
            if parent.right:
                queue.append(parent.right)
            while queue:
                node = queue.popleft()
                if node.data == sibling:
                    return parent
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)


if __name__ == '__main__':
    '''
    从根节点开始遍历,他的最左边和最右边分别为极大值和极小值
                  49
                 /  \
                38  65
               /   /  \
              13  52  76
               \  
               27 
    '''
    tree = BST()
    for num in [49, 38, 65, 76, 13, 27, 52]:  # 时间复杂度介于O(nlogn)和O(n^2),后者出现在序列已经有序的情况下
        tree.add(num)
    print(tree.find_common_parent_stack(38, 52))

    '''
    a,b,c,d,e,f=6,5,4,3,2,1
    F=(a+b)/((c-d)*e)*(-f) #后续遍历即可得到后缀表达式
                '*'
              /     \
            '/'     '-'
           /   \      \
         '+'    '*'    f
         / \    / \
        a   b  '-' e
               / \
              c   d
    root=Node('*',Node('/',Node('+',Node(a),Node(b)),Node('*',Node('-',Node(c),Node(d)),Node(e))),Node('-',None,Node(f)))
    def post_order_calc(root):
        operators={
            '+':lambda x,y:x+y,
            '-':lambda x,y:x-y,
            '*':lambda x,y:x*y,
            '/':lambda x,y:x/y,
        }
        if root:
            if isinstance(root.data,str):
                left_value=post_order_calc(root.left)
                right_value=post_order_calc(root.right)
                return operators[root.data](left_value,right_value)
            else:
                return root.data
        else:
            return 0
    '''
