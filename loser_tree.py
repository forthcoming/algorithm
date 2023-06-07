class LoserTree:
    def __init__(self, arr, key=lambda x, y: x > y):
        length = len(arr)
        self.key = key
        self.extreme_value = float("inf") if key(1, 0) else -float("inf")
        self.node = [length] * length  # 初始状态败者节点都指向leaves[-1]
        self.count = length
        self.leaves = arr + [-self.extreme_value]
        for index in range(length - 1, -1, -1):
            self.adjust(index)

    def adjust(self, index):
        parent = (self.count + index) >> 1
        while parent > 0:
            if self.key(self.leaves[index], self.leaves[self.node[parent]]):
                # 说明leaves[index]是败者,所以对它的父节点进行赋值,交换后index变成了优胜者
                self.node[parent], index = index, self.node[parent]
            parent >>= 1
        self.node[0] = index  # 循环结束后index一定是最后的优胜者

    def sort(self):
        if self.count <= 0:
            return
        while True:
            min_pos = self.node[0]
            min_value = self.leaves[min_pos]
            if min_value == self.extreme_value:
                break
            print(min_value)
            self.leaves[min_pos] = self.extreme_value
            self.adjust(min_pos)


if __name__ == "__main__":
    loser_tree = LoserTree([2, 4, 1, 0, 3, 7, 6])
    loser_tree.sort()
