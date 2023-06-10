#  常用于外排序,由数组构成的完全二叉树,每个节点保存败者下标
class LoserTree:
    def __init__(self, arr, key=lambda x, y: x > y):
        length = len(arr)
        self.key = key
        self.extreme_value = float("inf") if key(1, 0) else -float("inf")
        self.node = [length] * length  # 保存败者下标的完全二叉树,初始状态败者节点都指向leaves[-1]
        self.count = length
        self.leaves = arr + [-self.extreme_value]  # 这里必须取反,否则只会有最上方的根节点被赋值
        # for index in range(0, length):  # 待验证
        for index in range(length - 1, -1, -1):
            self.adjust(index)

    def adjust(self, index):
        """
        边看作其下面叶子结点的最小值; 非叶子结点等于其左右两条边的最大值
        当某个叶子结点值改变时,对应父边的值会改变,会影响到父节点值的改变,进而改变父节点的父边值改变,以此递归至根节点
        """
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
            extreme_pos = self.node[0]
            extreme_value = self.leaves[extreme_pos]
            if extreme_value == self.extreme_value:
                break
            print(extreme_value)
            self.leaves[extreme_pos] = self.extreme_value
            self.adjust(extreme_pos)


if __name__ == "__main__":
    '''
                       3
                       |
                       0
                  /         \
                 2           6
               /   \       /   \
              1     4     5     L0
              /\    /\    /\
             L1 L2 L3 L4 L5 L6           
    '''
    loser_tree = LoserTree([2, 4, 1, 0, 3, 7, 6])  # 外排序中每个归并段都是有序集合,这里相当于7个集合元素为1的归并段
    loser_tree.sort()
