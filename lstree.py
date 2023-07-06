class Node:
    def __init__(self, data=-1, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __str__(self):
        return f'data:{self.data}'


class Hamming:
    def __init__(self, depth=64):
        self.__root = Node(-1)
        self.__depth = depth  # signature's binary digits and binary_tree's depth(not including root node)

    def insert(self, signature, _id):
        root = self.__root
        for _ in range(self.__depth):  # the nodes generated from low to high
            if signature & 1:  # left-child:1,right-child:0
                if not root.left:
                    root.left = Node()
                root = root.left
            else:
                if not root.right:
                    root.right = Node()
                root = root.right
            signature >>= 1
        root.data = _id

    def find(self, signature, dis):
        result = []

        def _find(root, _signature, _dis, depth):
            if _dis >= 0:
                if depth == self.__depth:
                    result.append(root.data)
                    return
                bit = _signature & 1
                _signature >>= 1
                if bit:
                    if root.left:
                        _find(root.left, _signature, _dis, depth + 1)
                    if root.right:
                        _find(root.right, _signature, _dis - 1, depth + 1)
                else:
                    if root.left:
                        _find(root.left, _signature, _dis - 1, depth + 1)
                    if root.right:
                        _find(root.right, _signature, _dis, depth + 1)

        _find(self.__root, signature, dis, 0)
        return result


if __name__ == '__main__':
    lsh = Hamming(4)
    signatures = [0b0011, 0b1001, 0b1100, 0b0111]
    """
                  #
              /      \
              1       0
            /   \      \
           1     0      0
          / \     \     /
         1   0     0   1
          \   \    /  /
           0   0  1  1 

    """
    for index, each in enumerate(signatures):
        lsh.insert(each, index)
    print(lsh.find(0b1001, 2))
