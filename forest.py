from collections import deque

class Node:
    def __init__(self,data,left=None,sibling=None):
        self.data=data
        self.left=left
        self.sibling=sibling
    def __str__(self):
        return 'data:{}'.format(self.data)

class Forest:
    def __init__(self,root=None):
        self.__root=root

    def init(self):
        '''
              0            1              0
             / \         / | \         /     \
            2   3   +   4  5  6   =   2       1
                |       |  |           \     /
                7       8  9            3   4
                                        /  / \
                                       7  8   5
                                             / \
                                            9   6
        '''
        self.__root=Node(0,Node(2,sibling=Node(3,left=Node(7))),Node(1,Node(4,Node(8),Node(5,Node(9),Node(6)))))

    @property
    def root(self):
        return self.__root

    def create_forest(self,nodes=[('#',0),('#',1),(0,2),(0,3),(0,4),(1,5),(1,6),(2,7),(3,8),(6,9)]):
        '''                     
               0           1       
             / | \        / \       
            2  3  4  +   5   6 
            |  |             |       
            7  8             9                       
        '''
        queue=deque([Node(nodes[0][1])])
        self.__root=queue[0]
        index=1
        length=len(nodes)
        root=queue[0]
        while index<length and nodes[index][0]=='#':
            root.sibling=Node(nodes[index][1])
            root=root.sibling
            index+=1
        while queue and index<length:   
            node=queue.popleft()
            while node and index<length:
                if node.data!=nodes[index][0]:
                    node=node.sibling
                else:
                    if node.left:
                        r.sibling=Node(nodes[index][1])
                        r=r.sibling
                    else:
                        node.left=Node(nodes[index][1])
                        r=node.left
                        queue.append(r)  #如果要迭代调用其sibling,则队列里面只能存他的一个左孩子
                    index+=1

    def level_traverse(self):
        queue=deque([self.__root])
        while queue:
            node=queue.popleft()
            while node:
                print(node,end='\t')
                if node.left:
                    queue.append(node.left)
                node=node.sibling

    def root_first_traverse(self,root): #对应二叉树的前序遍历,以广义表形式打印树结构
        while root:
            print(root,end='')
            if root.left:
                print('(',end='')
                self.root_first_traverse(root.left)
                print(')',end='')
            root=root.sibling
            if root:
                print(',',end='')
        # if root:
        #     print(root,end='')
        #     if root.left:
        #         print('(',end='')
        #         self.root_first_traverse(root.left)
        #         print(')',end='')
        #     if root.sibling:
        #         print(',',end='')
        #         self.root_first_traverse(root.sibling)

    def root_last_traverse(self,root): #对应二叉树的中序遍历
        while root:
            if root.left:
                self.root_last_traverse(root.left)
            print(root,end='\t')
            root=root.sibling
        # if root:
        #     self.root_last_traverse(root.left)
        #     print(root,end='\t')
        #     self.root_last_traverse(root.sibling)

    def find_parent(self,key):  #所有的遍历方式都行,但最好不要用递归
        queue=deque([self.__root])
        while queue:
            node=queue.popleft()
            while node:
                child=node.left
                while child:
                    if child.data==key.data:
                        return node
                    child=child.sibling
                if node.left:
                    queue.append(node.left)
                node=node.sibling

    def max_depth(self,root):
        if root:
            return max(1+self.max_depth(root.left),self.max_depth(root.sibling))  #注意与二叉树最大深度的区别
        else:
            return 0

    def show_route(self,root,queue=deque()):  #当然如果用[]来当做栈使用也是可以的,树的叶子结点特征是左子树为空
        while root:
            queue.append(root)
            if root.left:
                self.show_route(root.left)
            else:
                ends=queue[-1]
                while True:
                    node=queue.popleft()
                    print(node,end=' ')
                    queue.append(node)
                    if node==ends:
                        break
                print()
            queue.pop()
            root=root.sibling

            
if __name__=='__main__':
    forest=Forest()
    forest.create_forest()
    forest.level_traverse()
    print(forest.max_depth(forest.root))
    forest.show_route(forest.root)
