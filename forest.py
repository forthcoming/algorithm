from collections import deque

'''
树与二叉树,森林与二叉树之间转换唯一确定,任意一颗树对应的二叉树根节点无右子树
树/森林的存储结构
1. 双亲链表---并查集
2. 孩子链表---图的一种存储方式（树也是一种特殊的图结构）
3. 二叉链表---按照二叉树的结构存储（推荐,可以利用二叉树的研究方法研究）
树/森林先序遍历对应二叉树先序遍历; 树/森林后序遍历对应二叉树中序遍历
二叉树最大子结构是根节点下的左右子树
森林最大子结构是每棵树根节点下的最左孩子(从左到右第一个孩子),最大子结构也可以是第一棵树根节点下的最左孩子和第一棵树后面的子树集合root.sibling
'''


class Node:
    def __init__(self, data, left=None, sibling=None):
        self.data = data
        self.left = left
        self.sibling = sibling

    def __str__(self):
        return f'{self.data}'


class Forest:
    def __init__(self, root=None):
        self.__root = root

    @property
    def root(self):
        return self.__root

    def create_forest(self, nodes):
        queue = deque([Node(nodes[0][1])])
        self.__root = queue[0]
        index = 1
        length = len(nodes)
        root = queue[0]
        while index < length and nodes[index][0] == '#':  # 将其兄弟连接起来,且队列只保存最左边的孩子节点
            root.sibling = Node(nodes[index][1])
            root = root.sibling
            index += 1
        while queue:
            node = queue.popleft()
            while node and index < length:
                if node.data != nodes[index][0]:
                    node = node.sibling
                else:
                    if node.left:
                        r.sibling = Node(nodes[index][1])
                        r = r.sibling
                    else:
                        node.left = Node(nodes[index][1])
                        r = node.left
                        queue.append(r)  # 队列只保存最左边的孩子节点,应为兄弟节点可以通过其遍历出来
                    index += 1

    def level_traverse(self):
        queue = deque([self.__root])
        while queue:
            node = queue.popleft()
            while node:
                print(node, end="\t")
                if node.left:
                    queue.append(node.left)
                node = node.sibling

    def root_last_traverse(self):
        def _root_last_traverse(root):
            while root:
                _root_last_traverse(root.left)
                print(root, end="\t")
                root = root.sibling

            # if root:  # 二叉树中序遍历
            #     _root_last_traverse(root.left)
            #     print(root, end='\t')
            #     _root_last_traverse(root.sibling)

        _root_last_traverse(self.__root)

    def root_first_traverse(self, root):  # 对应二叉树的前序遍历,以广义表形式打印树结构
        while root:
            print(root, end='')
            if root.left:
                print('(', end='')
                self.root_first_traverse(root.left)
                print(')', end='')
            root = root.sibling
            if root:
                print(',', end='')
        # if root:
        #     print(root, end='')
        #     if root.left:
        #         print('(', end='')
        #         self.root_first_traverse(root.left)
        #         print(')', end='')
        #     if root.sibling:
        #         print(',', end='')
        #         self.root_first_traverse(root.sibling)

    def find_parent(self, element):  # 层序遍历方式
        queue = deque([self.__root])
        while queue:
            node = queue.popleft()
            while node:
                child = node.left
                while child:
                    if child.data == element:
                        return node
                    child = child.sibling
                if node.left:
                    queue.append(node.left)
                node = node.sibling

    def max_depth(self, root):
        if root:
            return max(1 + self.max_depth(root.left), self.max_depth(root.sibling))  # 注意与二叉树最大深度的区别
        else:
            return 0

    def show_route(self, root, queue=deque()):  # 当然如果用[]来当做栈使用也是可以的,树的叶子结点特征是左子树为空
        while root:
            queue.append(root)
            if root.left:
                self.show_route(root.left)
            else:
                ends = queue[-1]
                while True:
                    node = queue.popleft()
                    print(node, end=' ')
                    queue.append(node)
                    if node == ends:
                        break
                print()
            queue.pop()
            root = root.sibling


if __name__ == '__main__':
    forest = Forest()
    forest.create_forest([('#', 0), ('#', 1), (0, 2), (0, 3), (0, 4), (1, 5), (1, 6), (2, 7), (3, 8), (6, 9)])
    """                     
           0           1                   0
         / | \        / \                /   \
        2  3  4  +   5   6     =        2     1
        |  |             |             / \    /
        7  8             9            7   3  5
                                         /\   \   
                                        8  4   6
                                               /
                                              9   
    """
    forest.level_traverse()
    print(forest.max_depth(forest.root))
    forest.root_last_traverse()
    print(forest.find_parent(4))
    # forest.show_route(forest.root)

