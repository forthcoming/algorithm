# 堆
class Heap:
    def __init__(self,heap=[],key=lambda x,y:x>y):  #默认构建小顶堆
        self.__heap=list(heap)
        self.length=len(heap)
        self.key=key                   
        self.buildHeap()   #构建过程时间复杂度是O(n)

    def isLeaf(self,pos):
        # return (self.length>>1)-1<pos<self.length
        return (pos<<1)+1>=self.length # 叶子结点无左孩子

    def leftChild(self,pos): #不存在则返回-1
        pos=(pos<<1)+1
        if 0<pos<self.length:
            return pos
        else:
            return -1

    def rightChild(self,pos):
        pos=(pos<<1)+2
        if 0<pos<self.length:
            return pos
        else:
            return -1

    def parent(self,pos):
        if 0<pos<self.length:
            return (pos-1)>>1
        else:
            return -1

    def traverse(self):
        print(self.__heap[:self.length])

    def pop(self,pos=0):  #pop,push原则是不能使__heap元素移位,时间复杂度是O(logn)
        assert self.length and 0<=pos<self.length
        self.length-=1
        value=self.__heap[pos]
        self.__heap[pos]=self.__heap[self.length]
        senior=self.parent(pos)
        if senior==-1 or self.key(self.__heap[pos],self.__heap[senior]):
            self.__shiftDown(pos,self.length-1)  #注意此处的shiftdown和shiftup是互斥的
        else:
            self.__shiftUp(pos)
        return value

    def push(self,value):  #时间复杂度是O(logn)
        self.length+=1
        try:
            self.__heap[self.length-1]=value
        except IndexError:
            self.__heap.append(value)
        self.__shiftUp(self.length-1)

    def __shiftDown(self,starts,ends):  #为了排序而多加了一个ends参数
        root=self.__heap[starts]
        left=self.leftChild(starts)
        while left!=-1 and left<=ends:
            if left<ends and self.key(self.__heap[left],self.__heap[left+1]):
                left+=1
            if self.key(root,self.__heap[left]):
                self.__heap[starts]=self.__heap[left]
                starts=left
                left=self.leftChild(starts)
            else:
                break
        self.__heap[starts]=root

    def __shiftUp(self,ends):
        root=self.__heap[ends]
        senior=self.parent(ends)
        while senior!=-1 and self.key(self.__heap[senior],root):
            self.__heap[ends]=self.__heap[senior]
            ends=senior
            senior=self.parent(ends)                 
        self.__heap[ends]=root

    def buildHeap(self):  #建堆的时间复杂度是O(n)
        #for ends in range(0,self.length):  #自上而下构建堆
            #self.__shiftUp(ends)
        for starts in range((self.length>>1)-1,-1,-1):  #自下而上构建堆,只需要从非叶子节点开始构建
            self.__shiftDown(starts,self.length-1)

    def sort(self):  #堆排序不稳定
        self.buildHeap()
        for ends in range(self.length-1,0,-1):
            self.__heap[0],self.__heap[ends]=self.__heap[ends],self.__heap[0]
            self.__shiftDown(0,ends-1)

    def countLeftChild(self,size):  #计算根节点的左孩子个数
        import math
        height=int(math.log2(size+1))
        X=size+1-2**height
        X=min(X,2**(height-1))
        return X+2**(height-1)-1
    
    def createCompleteSearchTree(self):  #跟堆相关的一道题目(不属于堆结构)
        self.sort()
        tree=[None]*self.length
        def __solve(root,start,size):
            if size:
                lchilds=self.countLeftChild(size)
                tree[root]=self.__heap[start+lchilds]
                left=(root<<1)+1
                right=left+1
                __solve(left,start,lchilds)
                __solve(right,start+lchilds+1,size-1-lchilds)
        __solve(0,0,self.length)
        return tree

if __name__=='__main__':
    heap=Heap([3,54,64,34,24,2,24,33],key=lambda x,y:x<y)
    heap.push(4)
    heap.push(21)
    heap.traverse()
    print(heap.pop(2))
    heap.sort()
    heap.traverse()
