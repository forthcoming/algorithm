class LoserTree:
    def __init__(self, arr):
        length = len(arr)
        self.max_value = float("inf")
        self.node = [length] * length
        self.count = length
        self.leaves = arr + [-self.max_value]
        for index in range(length - 1, -1, -1):
            self.adjust(index)

    def adjust(self, index):
        parent = (self.count + index) >> 1
        while parent > 0:
            if self.leaves[index] > self.leaves[self.node[parent]]:
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
            if min_value == self.max_value:
                break
            print(min_value)
            self.leaves[min_pos] = self.max_value
            self.adjust(min_pos)


if __name__ == "__main__":
    loser_tree = LoserTree([2, 4, 1, 0, 3, 7, 6])
    loser_tree.sort()
