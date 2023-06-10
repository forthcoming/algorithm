from collections import deque

"""
假定在𝑇𝑟𝑖𝑒树上遍历至𝑥得到的字符串为𝑤𝑜𝑟𝑑𝑥,且𝑓𝑎𝑖𝑙[𝑥]指向的是𝑦,则𝑤𝑜𝑟𝑑𝑦是𝑤𝑜𝑟𝑑𝑥在这棵树上所能匹配到的最长后缀,可能对应着另一个模式串
AC自动机是一种有限状态自动机,当只有一个模式串时会退化为kmp算法
由于匹配过程中匹配串text的字符指针只会向前进,且每次遇到不匹配时都是重新找最长后缀,所以一定会不重且不漏找到所有结果
"""


class Node:
    def __init__(self):
        self.children = {}
        self.fail = None  # 失败指针
        self.output = []  # 所有后缀串中属于模式串的长度
        # self.char       # 当前节点字符(本算法不需要)
        # self.is_end     # 根节点到当前节点是否是模式串(本算法不需要)


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
        root.output = [len(pattern)]

    def build_fail(self):
        queue = deque([self.__root])
        while queue:
            node = queue.popleft()
            for char, son in node.children.items():
                next_node = node.fail
                while next_node:
                    if char in next_node.children:
                        son.fail = next_node.children[char]
                        son.output += son.fail.output
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
                for start in son.output:
                    result.append((end - start, end))
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
