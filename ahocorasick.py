from collections import deque

"""
假定在𝑇𝑟𝑖𝑒树上遍历至𝑥得到的字符串为𝑤𝑜𝑟𝑑𝑥,且𝑓𝑎𝑖𝑙[𝑥]指向的是𝑦,则𝑤𝑜𝑟𝑑𝑦是𝑤𝑜𝑟𝑑𝑥在这棵树上所能匹配到的最长后缀,可能对应着另一个模式串
AC自动机是一种有限状态自动机,当只有一个模式串时会退化为kmp算法
"""


class Node:
    def __init__(self):
        self.children = {}
        self.fail = None  # 失败指针
        self.length = 0


class AcAutomaton:
    def __init__(self, patterns):
        self.__root = Node()
        for pattern in patterns:
            self.insert(pattern)
        self.build_fail()

    def insert(self, pattern):
        root = self.__root
        for char in pattern:
            root = root.children.setdefault(char, Node())
        root.length = len(pattern)

    def build_fail(self):
        queue = deque([self.__root])
        while queue:
            node = queue.popleft()
            for char, son in node.children.items():
                next_node = node.fail
                while next_node:
                    if char in next_node.children:
                        son.fail = next_node.children[char]
                        break
                    next_node = next_node.fail
                else:
                    son.fail = self.__root
                queue.append(son)

    def search(self, text):
        result = []
        parent = self.__root
        for end, char in enumerate(text, 1):
            son = parent.children.get(char)
            while (not son) and (parent != self.__root):
                parent = parent.fail
                son = parent.children.get(char)
            if son:
                parent = son
                while son:
                    if son.length:
                        result.append((end - son.length, end))
                    son = son.fail
        return result


if __name__ == '__main__':
    """
                #
             /  |  \
            C   D   H
           /\   |   |
          C  D  H   Y
          |  |
          D  H
          |
          H  
    """
    ac_automaton = AcAutomaton(["CD", "CDH", "CCDH", "HY", "DH", "CCD"])
    print(ac_automaton.search('GGCDHCCDHY'))
