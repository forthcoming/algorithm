# to be continued

from collections import deque

class Node:
    def __init__(self,data,left=None,right=None):
        self.data=data
        self.left=left
        self.right=right

    def __str__(self):
        return str(self.data)

class BinaryTree:   # 度为0的个数=度为2的个数+1
    def __init__(self,root=None):
        self.__root=root
        
    @property
    def root(self):
        return self.__root

    def post_order_copy(self):  # 思想类似于归并
        def _post_order_copy(node):
            if node:
                left=_post_order_copy(node.left)
                right=_post_order_copy(node.right)
                return Node(node.data,left,right)
        return _post_order_copy(self.__root)
    
    def is_same(self,another):
        def _is_same(first,second):
            if first and second:
                return (first.data==second.data) and _is_same(first.left,second.left) and _is_same(first.right,second.right)
            else:
                return not(first or second)
        return _is_same(self.__root,another.__root)
    
    def pre_order(self):
        def _pre_order(root):
            print(root,end='\t')
            # root.left,root.right=root.right,root.left #反转二叉树
            if root.left:  #减小递归深度
                _pre_order(root.left)
            if root.right:
                _pre_order(root.right)
        if self.__root:
            _pre_order(self.__root)

    def pre_order_stack(self):
        stack=[]
        root=self.__root
        while root or stack:
            if root:
                print(root)
                if root.right:  #提高效率
                    stack.append(root.right) #也可以让root入栈,出栈时令root=stack.pop().right
                root=root.left
                # root.left,root.right=root.right,root.left  # 反转二叉树
                # root=root.right 
            else:
                root=stack.pop()

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
        stack=[]
        root=self.__root
        while root or stack:
            if root:
                stack.append(root)
                root=root.left
            else:
                root=stack.pop()
                print(root)
                root=root.right

    def post_order(self):
        def _post_order(root):
            if root.left:
                _post_order(root.left)
            if root.right:
                _post_order(root.right)
            print(root)
        if self.__root:
            _post_order(self.__root)
    
    def BFS(self):
        count=0   # 结点个数
        if self.__root:
            queue=deque([self.__root])
            while queue:
                node=queue.popleft()
                count+=1
                print(node)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return count   
    
    def find_path(self):  #涉及root => leaf路径问题，一律是先序遍历
        path=[]
        def _find_path(root):
            path.append(root.data)
            if root.left or root.right:
                if root.left:
                    _find_path(root.left)
                if root.right:
                    _find_path(root.right)
            else:
                print(path)
            path.pop()
        if self.__root:
            _find_path(self.__root)
            
    def find_path_stack(self):  # all path from root to leaf
        root=self.__root
        stack=[]
        path=[]
        while root or stack:
            if root:
                path.append(root.data)
                if root.right:
                    stack.append(root) # 这里不能让其右孩子入栈
                elif not root.left:
                    print(path)
                root=root.left
            else:
                root=stack.pop()
                if root.left:  # 一定要判断
                    node=path.pop()
                    l_data=root.left.data
                    while l_data!=node:
                        node=path.pop()
                root=root.right
    
    def BFS_nth_depth(self,num): #可以求二叉树最大宽度
        if self.__root:
            depth=1
            queue=deque((self.__root,depth))
            while queue:
                node=queue.popleft()
                if isinstance(node,int):
                    if queue:
                        depth+=1
                        queue.append(depth)
                    else:
                        return -1
                else:
                    if node.data==num:
                        return depth
                    if node.left:
                        queue.append(node.left)
                    if node.right:
                        queue.append(node.right)
        else:
            return -1
       
        # 低效版
        # depth=-1
        # def _find(root,level):
        #     nonlocal depth
        #     if root.data==num:
        #         depth=level
        #     else:
        #         if root.left:
        #             _find(root.left,level+1)
        #         if root.right:
        #             _find(root.right,level+1)
        # if self.__root:
        #     _find(self.__root,1)
        # return depth
        
    def node_num(self,node):  # 各种遍历都可以计算结点个数
        if node:
            return self.node_num(node=node.left)+self.node_num(node=node.right)+1
        else:
            return 0

    def max_depth(self):  # BFS也可以计算最大深度
        def _max_depth(node):
            if node:
                return max(_max_depth(node.left),_max_depth(node.right))+1
            else:
                return 0
        return _max_depth(self.__root)
        
        # depth=0
        # def _max_depth(node,level):
        #     nonlocal depth
        #     if level>depth:
        #         depth=level
        #     if node.left:
        #         _max_depth(node.left,level+1)
        #     if node.right:
        #         _max_depth(node.right,level+1)
        # if self.__root:
        #     _max_depth(self.__root,1)
        # return depth
        
    def find_parent(self,child,sibling):   #寻找最低公共父节点
        # 还有个思路，若child和sibling分别在当前节点的左右子树上,则当前节点即为最低公共父节点,否则递归调用其左子树或右子树
        root=self.__root
        stack=[]
        path=[]
        while root or stack:
            if root:
                if root.data==child:   #找到了child,则查看child的所有父节点中哪个也属于sibling父节点即可
                    return __class__.check(path,sibling)
                path.append(root)
                if root.right:
                    stack.append(root)
                root=root.left
            else:
                root=stack.pop()
                if root.left:
                    node=path.pop()
                    while root.left!=node:
                        node=path.pop()
                root=root.right 

    def check(path,sibling):
        queue=deque()
        while path:
            parent=path.pop()
            if parent.left:
                queue.append(parent.left)
            if parent.right:
                queue.append(parent.right)
            while queue:
                node=queue.popleft()
                if node.data==sibling:
                    return parent
                if node.left:
                    queue.append(node.left)
                if node.right:
                   queue.append(node.right)      
                
if __name__=='__main__':
    tree=BinaryTree()
    tree.pre_order()
    
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
    
