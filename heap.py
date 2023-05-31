# 堆
import math


def count_left_child(size):  # 计算根节点的左孩子个数
    height = int(math.log2(size + 1))
    X = size + 1 - 2 ** height
    X = min(X, 2 ** (height - 1))
    return X + 2 ** (height - 1) - 1


class Heap:
    def __init__(self, arr, key=lambda x, y: x > y):  # 默认构建小顶堆
        self.__heap = list(arr)
        self.length = len(arr)
        self.key = key
        self.build_heap()  # 构建过程时间复杂度是O(n)

    def is_leaf(self, pos):
        # return (self.length>>1)-1<pos<self.length
        return (pos << 1) + 1 >= self.length  # 叶子结点无左孩子

    def left_child(self, pos):  # 不存在则返回-1
        pos = (pos << 1) + 1
        if 0 < pos < self.length:
            return pos
        else:
            return -1

    def right_child(self, pos):
        pos = (pos << 1) + 2
        if 0 < pos < self.length:
            return pos
        else:
            return -1

    def parent(self, pos):
        if 0 < pos < self.length:
            return (pos - 1) >> 1
        else:
            return -1

    def traverse(self):
        print(self.__heap[:self.length])

    def pop(self, pos=0):  # pop,push原则是不能使__heap元素移位,时间复杂度是O(logn)
        assert self.length and 0 <= pos < self.length
        self.length -= 1
        value = self.__heap[pos]
        self.__heap[pos] = self.__heap[self.length]
        senior = self.parent(pos)
        if senior == -1 or self.key(self.__heap[pos], self.__heap[senior]):
            self.__shift_down(pos, self.length - 1)  # 注意此处的shift_down和shift_up是互斥的
        else:
            self.__shift_up(pos)
        return value

    def push(self, value):  # 时间复杂度是O(logn)
        self.length += 1
        try:
            self.__heap[self.length - 1] = value
        except IndexError:
            self.__heap.append(value)
        self.__shift_up(self.length - 1)

    def __shift_down(self, starts, ends):  # 为了排序而多加了一个ends参数
        root = self.__heap[starts]
        left = self.left_child(starts)
        while left != -1 and left <= ends:
            if left < ends and self.key(self.__heap[left], self.__heap[left + 1]):
                left += 1
            if self.key(root, self.__heap[left]):
                self.__heap[starts] = self.__heap[left]
                starts = left
                left = self.left_child(starts)
            else:
                break
        self.__heap[starts] = root

    def __shift_up(self, ends):
        root = self.__heap[ends]
        senior = self.parent(ends)
        while senior != -1 and self.key(self.__heap[senior], root):
            self.__heap[ends] = self.__heap[senior]
            ends = senior
            senior = self.parent(ends)
        self.__heap[ends] = root

    def build_heap(self):  # 建堆的时间复杂度是O(n)
        # for ends in range(0,self.length):  #自上而下构建堆
        # self.__shiftUp(ends)
        for starts in range((self.length >> 1) - 1, -1, -1):  # 自下而上构建堆,只需要从非叶子节点开始构建
            self.__shift_down(starts, self.length - 1)

    def sort(self):  # 堆排序不稳定
        self.build_heap()
        for ends in range(self.length - 1, 0, -1):
            self.__heap[0], self.__heap[ends] = self.__heap[ends], self.__heap[0]
            self.__shift_down(0, ends - 1)

    def create_complete_search_tree(self):  # 跟堆相关的一道题目(不属于堆结构)
        self.sort()
        tree = [None] * self.length

        def __solve(root, start, size):
            if size:
                l_children = count_left_child(size)
                tree[root] = self.__heap[start + l_children]
                left = (root << 1) + 1
                right = left + 1
                __solve(left, start, l_children)
                __solve(right, start + l_children + 1, size - 1 - l_children)

        __solve(0, 0, self.length)
        return tree


if __name__ == '__main__':
    heap = Heap([3, 54, 64, 34, 24, 2, 24, 33], key=lambda x, y: x < y)
    heap.push(4)
    heap.push(21)
    heap.traverse()
    print(heap.pop(2))
    heap.sort()
    heap.traverse()
