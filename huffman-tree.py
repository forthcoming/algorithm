class Node:
    def __init__(self,weight,left=None,right=None):
        self.weight=weight
        self.left=left
        self.right=right
    def __lt__(self,other):
        return self.weight<other.weight

class HuffmanTree:
    def __init__(self,words='github.com/xxoome'):
        from collections import defaultdict
        from heapq import heapify,heappush,heappop
        weights=defaultdict(int)  #or defaultdict(lambda:0),由于每次程序执行时字典值的顺序都不一样导致tree不一样,但不影响结果的正确性
        for each in words:
            weights[each]+=1
        heap=[Node(weights[each]) for each in weights]
        heapify(heap)
        while len(heap)>1:
            lchild=heappop(heap)
            rchild=heappop(heap)
            heappush(heap,Node(lchild.weight+rchild.weight,lchild,rchild))
            __class__.check_leaf(lchild,weights)
            __class__.check_leaf(rchild,weights)
        self.__root=heappop(heap) if heap else None
        if weights:
            __class__.check_leaf(self.__root,weights)
        self.words=words

    @staticmethod
    def check_leaf(node,weights):
        if not node.left: #叶子节点
            for each in weights:
                if weights[each]==node.weight:
                    node.weight=each
                    weights.pop(each)
                    break

    def encode(self):
        encoding={}
        def _encode(root,stack=[]):
            if root.left:  #哈夫曼树除叶子节点外度都是2
                stack.append('0')
                _encode(root.left)
                stack.pop()
                stack.append('1')
                _encode(root.right)
                stack.pop()
            else:
                encoding[root.weight]=''.join(stack)
        # def _encode(root,flag=None,stack=[]):
        #     stack.append(flag)
        #     if root.left:
        #         _encode(root.left,flag='0')
        #         _encode(root.right,flag='1')
        #     else:
        #         encoding[root.weight]=''.join(stack[1:])
        #     stack.pop()
        if self.__root.left:
            _encode(self.__root)
            return ''.join(encoding[i] for i in self.words)
        else:
            return '0'*len(self.words)

    def decode(self,password):
        root=self.__root
        if root.left:
            for i in password:
                root=root.right if int(i) else root.left
                if not root.left:
                    print(root.weight,end='')
                    root=self.__root
        else:
            print(root.weight*len(password))

            
if __name__=='__main__':
    huffman=HuffmanTree()
    password=huffman.encode()
    print(password)
    huffman.decode(password)
