# 并查集
class UnionFindSet:
    def __init__(self,sets=[-4,0,1,1,-6,4,4,5,7,5]): #非负数指向父节点的下标,负数n代表该根节点下有|n|个孩子
        '''
             0       4
             |      / \            
             1     5   6    
            / \   / \              
           2   3  7  9
                  |
                  8
        '''
        self.__set=sets

    def union(self,i,j):
        I=self.find(i)
        J=self.find(j)
        if I!=J:
            if self.__set[I]>self.__set[J]:  #__set[i]集合元素个数更少(将小结果集追加到大结果集中)
                self.__set[J]+=self.__set[I]
                self.__set[I]=J
            else:
                self.__set[I]+=self.__set[J]
                self.__set[J]=I
            return True
        return False

    def find(self,i):  #路径压缩算法(查找父亲的同时减小树的深度)
        if self.__set[i]>=0:
            self.__set[i]=self.find(self.__set[i])
            return self.__set[i]
        return i
    
    # def find(self,i):  #迭代版路径压缩算法(查找父亲的同时减小树的深度)
    #     j=i
    #     while self.__set[j]>=0:
    #         j=self.__set[j]
    #     while self.__set[i]>=0:
    #         k=self.__set[i]
    #         self.__set[i]=j
    #         i=k
    #     return j
