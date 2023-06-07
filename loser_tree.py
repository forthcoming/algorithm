class LoserTree:
    def __init__(self, arr):
        length = len(arr)
        self.node = [length] * length
        self.count = length
        self.leaves = arr + [-float("inf")]
        for index in range(length - 1, -1, -1):
            self.adjust(index)

    def adjust(self, index):
        parent = (self.count + index) >> 1
        while parent > 0:
            if self.leaves[index] > self.leaves[self.node[parent]]:
                # 值大者败
                # 进入此条件，说明leaves[index]是败者。所以对它的父节点进行赋值
                self.node[parent], index = index, self.node[parent]
                # 交换后index变成了优胜者
                # 求出parent的parent,进入下一轮循环
            parent >>= 1
        self.node[0] = index  # 循环结束后index一定是最后的优胜者

    def sort(self):
        if self.count <= 0:
            return
        max_value = float("inf")
        while True:
            min_pos = self.node[0]
            min_value = self.leaves[min_pos]
            if min_value == max_value:
                break
            print(min_value)
            self.leaves[min_pos] = max_value
            self.adjust(min_pos)


if __name__ == "__main__":
    loser_tree = LoserTree([2, 4, 1, 0, 3, 7, 6])
    loser_tree.sort()
