from collections import deque


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
    ac_automaton = AcAutomaton(["CD", "CDH", "CCDH", "HY", "DH", "CCD"])
    print(ac_automaton.search('GGCDHCCDHY'))
