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

    def init(self,calc=False):  # just for test
        a,b,c,d,e,f=6,5,4,3,2,1
        F=(a+b)/((c-d)*e)*(-f) #后续遍历即可得到后缀表达式
        if calc:
            '''         '*'
                      /     \
                    '/'     '-'
                   /   \      \
                 '+'    '*'    f
                 / \    / \
                a   b  '-' e
                       / \
                      c   d
            '''
            self.__root=Node('*',Node('/',Node('+',Node(a),Node(b)),Node('*',Node('-',Node(c),Node(d)),Node(e))),Node('-',None,Node(f)))
        else:
            '''
                      0
                    /   \
                   1     2
                  / \     \
                 3   4     5
                /         / \
               6         7   8
                        / \
                       9  10
            '''
            self.__root=Node(0,Node(1,Node(3,Node(6)),Node(4)),Node(2,None,Node(5,Node(7,Node(9),Node(10)),Node(8))))

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

if __name__=='__main__':
    tree=BinaryTree()
    tree.init()
    tree.pre_order()
