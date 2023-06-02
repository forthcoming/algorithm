# 并查集
class DisjointSets:
    def __init__(self, sets):  # 非负数指向父节点的下标,负数n代表该根节点下有|n|个孩子
        self.__set = sets

    def find(self, i):  # 迭代版路径压缩算法
        root = i
        while self.__set[root] >= 0:
            root = self.__set[root]
        while root != i:
            k = self.__set[i]
            self.__set[i] = root
            i = k
        return root

    def rec_find(self, i):  # 路径压缩算法(查找父亲的同时减小树的深度,路径上的子节点在查找完一遍后通通指向父节点)
        if self.__set[i] >= 0:
            self.__set[i] = self.rec_find(self.__set[i])
            return self.__set[i]
        return i

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.__set[root_i] > self.__set[root_j]:  # __set[i]集合元素个数更少(将小结果集追加到大结果集中)
                self.__set[root_j] += self.__set[root_i]
                self.__set[root_i] = root_j
            else:
                self.__set[root_i] += self.__set[root_j]
                self.__set[root_j] = root_i
            return True
        return False


if __name__ == "__main__":
    """
         0       4
         |      / \
         1     5   6
        / \   / \
       2   3  7  9
              |
              8
    """
    ufs = DisjointSets([-4, 0, 1, 1, -6, 4, 4, 5, 7, 5])
    print(ufs.find(7))
