import math


class Heap:
    # 所有父节点优先级高于子节点,由数组构成的完全二叉树
    # 堆是变长的数据结构,败者树是定长数据结构
    # 堆节点记录的是元素值,败者树节点记录的是叶子结点对应元素的下标
    # 如果是有限长度数组排序,用堆更优,应为不用额外存储叶子结点且堆不断变小,如果是归并段排序用败者树,应为调整树结构只需要跟父节点比较即可
    def __init__(self, arr, key=lambda x, y: x > y):  # 默认构建小顶堆
        self.__heap = list(arr)
        self.length = len(arr)
        self.key = key
        self.build_heap()  # 构建过程时间复杂度是O(n)

    def is_leaf(self, pos):
        """
        由于叶子结点无左孩子可得: self.length <= 2*pos + 1 即 self.length/2 - 0.5 <= pos
        又因为self.length,pos都是整数,所以化简得: (self.length >> 1) - 1 < pos
        """
        return (self.length >> 1) - 1 < pos < self.length

    def left_child(self, pos):  # 不存在则返回-1
        l_child_pos = (pos << 1) + 1
        if 0 < l_child_pos < self.length:
            return l_child_pos
        else:
            return -1

    def right_child(self, pos):
        r_child_pos = (pos << 1) + 2
        if 0 < r_child_pos < self.length:
            return r_child_pos
        else:
            return -1

    def top(self):
        if self.length > 0:
            return self.__heap[0]

    def parent(self, pos):
        if 0 < pos < self.length:
            return (pos - 1) >> 1
        else:
            return -1

    def traverse(self):
        print(self.__heap[:self.length])  # 应为会有pop操作导致__heap变小

    def __shift_up(self, end):
        # 与败者树adjust有点像,注意区别
        root = self.__heap[end]
        senior = self.parent(end)
        while senior != -1 and self.key(self.__heap[senior], root):
            self.__heap[end] = self.__heap[senior]
            end = senior
            senior = self.parent(end)
        self.__heap[end] = root

    def __shift_down(self, parent):
        end = self.length - 1
        root = self.__heap[parent]
        left = self.left_child(parent)
        while left != -1:
            # 如果有右孩子且右孩子优先级比左孩子高
            if left < end and self.key(self.__heap[left], self.__heap[left + 1]):
                left += 1
            if self.key(root, self.__heap[left]):
                self.__heap[parent] = self.__heap[left]
                parent = left
                left = self.left_child(parent)
            else:
                break
        self.__heap[parent] = root  # 不能放在else下面,应为有可能没有左孩子进不去while循环

    def __shift_down_sort(self, parent, end):
        root = self.__heap[parent]
        left = self.left_child(parent)
        while left != -1 and left <= end:  # 注意这里要判断left<=end
            if left < end and self.key(self.__heap[left], self.__heap[left + 1]):
                left += 1
            if self.key(root, self.__heap[left]):
                self.__heap[parent] = self.__heap[left]
                parent = left
                left = self.left_child(parent)
            else:
                break
        self.__heap[parent] = root

    def build_heap(self):  # 建堆的时间复杂度是O(n),重要!
        # for end in range(0, self.length):  # 自上而下构建堆
        #     self.__shift_up(end)
        for start in range((self.length >> 1) - 1, -1, -1):  # 自下而上构建堆,只需要从非叶子节点开始构建
            self.__shift_down(start)

    def push(self, value):  # 时间复杂度是O(logn)
        self.length += 1
        try:
            self.__heap[self.length - 1] = value
        except IndexError:
            self.__heap.append(value)
        self.__shift_up(self.length - 1)

    def pop(self, pos=0):  # pop,push原则是不能使__heap元素移位,时间复杂度是O(logn)
        assert 0 <= pos < self.length
        self.length -= 1
        value = self.__heap[pos]
        self.__heap[pos] = self.__heap[self.length]
        senior = self.parent(pos)
        if senior == -1 or self.key(self.__heap[pos], self.__heap[senior]):
            self.__shift_down(pos)  # 注意此处的shift_down和shift_up是互斥的
        else:
            self.__shift_up(pos)
        return value

    def sort(self):  # 堆排序不稳定
        self.build_heap()
        for end in range(self.length - 1, 0, -1):
            self.__heap[0], self.__heap[end] = self.__heap[end], self.__heap[0]
            self.__shift_down_sort(0, end - 1)

    @staticmethod
    def count_left_child(size):  # 计算根节点的左孩子个数
        height = int(math.log2(size + 1))
        X = size + 1 - 2 ** height
        X = min(X, 2 ** (height - 1))
        return X + 2 ** (height - 1) - 1

    def create_complete_search_tree(self):  # 跟堆相关的一道题目(不属于堆结构)
        self.sort()
        tree = [None] * self.length

        def __solve(root, start, size):
            if size:
                l_children = Heap.count_left_child(size)
                tree[root] = self.__heap[start + l_children]
                left = (root << 1) + 1
                right = left + 1
                __solve(left, start, l_children)
                __solve(right, start + l_children + 1, size - 1 - l_children)

        __solve(0, 0, self.length)
        return tree


if __name__ == '__main__':
    heap = Heap([3, 54, 64, 34, 24, 2, 24, 33, 45], key=lambda x, y: x > y)
    heap.push(4)
    heap.push(21)
    heap.traverse()
    print(heap.pop(2))
    heap.sort()
    heap.traverse()
