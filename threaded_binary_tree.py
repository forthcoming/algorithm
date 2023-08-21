LINK = 0  # 指针
Thread = 1  # 线索

'''
注: 前序和后序线索二叉树在找节点的前驱和后驱时会比较麻烦(需要三叉链表,其中一个指针指向其双亲),所以线索二叉树一般都是中序线索二叉树
在后序线索二叉树中,查找指定结点*p的前趋结点:
*p的左子树为空,则p->lchild是前趋线索,指示其前趋结点
*p的左子树非空,*p的右子树非空,*p的右孩子必是其前趋(A的前趋是E)
*p的左子树非空,*p右子树为空时,*p的前趋必是其左孩子(E的前趋是F)
在后序线索二叉树中,查找指定结点*p的后继结点: 
*p是根,则*p是该二叉树后序遍历过程中最后一个访问到的结点,*p的后继为空
*p是其双亲的右孩子,则*p的后继结点就是其双亲结点(E的后继是A)
*p是其双亲的左孩子,但*P无右兄弟,*p的后续结点是其双亲结点(F的后继是E)
*p是其双亲的左孩子,但*p有右兄弟,则*p的后继是其双亲的右子树中第一个后序遍历到的结点,它是该子树中"最左下的叶结点"(B的后继是H)
'''


class Node:
    def __init__(self, data, left=None, right=None, l_flag=LINK, r_flag=LINK):
        self.data = data
        self.left = left
        self.right = right
        self.LFlag = l_flag
        self.RFlag = r_flag

    def __str__(self):
        return 'data:{}'.format(self.data)


class ThreadBinaryTree:
    def __init__(self, root=None):
        self.__root = root
        self.pre = None

    def init(self):
        """
                      0
                    /   \
                   1     2
                  / \     \
                 3   4     5
                /         / \
               6         7   8
                        / \
                       9  10
        """
        self.__root = Node(0, Node(1, Node(3, Node(6)), Node(4)),
                           Node(2, None, Node(5, Node(7, Node(9), Node(10)), Node(8))))

    def ldr_threading(self, root):
        if root:
            self.ldr_threading(root.left)
            if not root.left:
                root.LFlag = Thread
                root.left = self.pre
            if not self.pre.right:
                self.pre.RFlag = Thread
                self.pre.right = root
            self.pre = root
            self.ldr_threading(root.right)

    def threading(self):
        self.thr_root = Node(data=None, l_flag=LINK, r_flag=Thread)
        self.thr_root.right = self.thr_root
        if self.__root:
            self.thr_root.left = self.__root
            self.pre = self.thr_root
            self.ldr_threading(self.__root)
            self.pre.right = self.thr_root
            self.pre.RFlag = Thread
            self.thr_root.right = self.pre
        else:
            self.thr_root.left = self.thr_root

    def threading_traverse(self):
        cur = self.thr_root.left
        while cur != self.thr_root:
            while cur.LFlag == LINK:
                cur = cur.left
            print(cur)
            while cur.RFlag == Thread and cur.right != self.thr_root:
                cur = cur.right
                print(cur)
            cur = cur.right

    @staticmethod
    def precursor_node(pointer):
        if pointer.LFlag == Thread:
            return pointer.left
        else:
            pointer = pointer.left  # 找左子树最后访问的节点
            while pointer.RFlag == LINK:
                pointer = pointer.right
            return pointer

    @staticmethod
    def successor_node(pointer):
        if pointer.RFlag == Thread:
            return pointer.right
        else:
            pointer = pointer.right  # 找右子树最先访问的节点
            while pointer.LFlag == LINK:
                pointer = pointer.left
            return pointer


if __name__ == '__main__':
    tree = ThreadBinaryTree()
    tree.init()
    tree.threading()
    tree.threading_traverse()
