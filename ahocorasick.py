from collections import deque


class Node:
    def __init__(self):
        self.children = {}
        self.fail = None  # 失败指针
        self.is_word = False
        self.length = 0


class AcAutomaton:
    def __init__(self, patterns):
        self.__root = Node()
        for pattern in patterns:
            self.insert(pattern)

    def insert(self, pattern):
        root = self.__root
        for char in pattern:
            root = root.children.setdefault(char, Node())
        root.is_word = True
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

    def search1(self, text):
        result = []
        start = 0
        for index, char in enumerate(text):
            end = index
            parent = self.__root
            while char in parent.children:
                if parent == self.__root:
                    start = index
                parent = parent.children[char]
                if parent.is_word:
                    result.append((start, end + 1))
                if parent.children and end + 1 < len(text):
                    end += 1
                    char = text[end]
                else:
                    break
                while (char not in parent.children) and (parent != self.__root):
                    parent = parent.fail
                    start += 1
                if parent == self.__root:
                    break
        return result


if __name__ == '__main__':
    ah = AcAutomaton(["CD", "CDH", "CCDH", "HY", 'DH', 'CCD'])
    ah.build_fail()
    print(ah.search('GGCDHCCDHY'))
    print(ah.search1('GGCDHCCDHY'))

# import os
# import pickle
# import codecs
#
#
# class TrieNode:
#     def __init__(self):
#         self.success = dict()  # 转移表
#         self.failure = None  # 错误表
#         self.emits = set()  # 输出表
#
#
# class CreateAcAutomaton(object):
#
#     def __init__(self, patterns, save_path="  "):
#         """
#         :param patterns:  模式串列表
#         :param save_path:   AC自动机持久化位置
#         """
#         self._savePath = save_path.strip()
#         assert isinstance(self._savePath, str) and self._savePath != ""
#         self._patterns = patterns
#         if os.path.exists(self._savePath):
#             self._root = self.__load_corasick()
#         else:
#             self._root = TrieNode()
#             self.__insert_node()
#             self.__create_fail_path()
#             self.__save_corasick()
#
#     def __insert_node(self):
#         """
#         Create Trie
#         """
#         for pattern in self._patterns:
#             line = self._root
#             for character in pattern:
#                 line = line.success.setdefault(character, TrieNode())
#             line.emits.add(pattern)
#
#     def __create_fail_path(self):
#         """
#         Create Fail Path
#         """
#         my_queue = list()
#         for node in self._root.success.values():
#             node.failure = self._root
#             my_queue.append(node)
#         while len(my_queue) > 0:
#             gone_node = my_queue.pop(0)
#             for k, v in gone_node.success.items():
#                 my_queue.append(v)
#                 parent_failure = gone_node.failure
#
#                 while parent_failure and k not in parent_failure.success.keys():
#                     parent_failure = parent_failure.failure
#                 v.failure = parent_failure.success[k] if parent_failure else self._root
#                 if v.failure.emits:
#                     v.emits = v.emits.union(v.failure.emits)
#
#     def __save_corasick(self):
#         with codecs.open(self._savePath, "wb") as f:
#             pickle.dump(self._root, f)
#
#     def __load_corasick(self):
#         with codecs.open(self._savePath, "rb") as f:
#             return pickle.load(f)
#
#     def search(self, context):
#         """"""
#         search_result = list()
#         search_node = self._root
#         for char in context:
#             while search_node and char not in search_node.success.keys():
#                 search_node = search_node.failure
#             if not search_node:
#                 search_node = self._root
#                 continue
#             search_node = search_node.success[char]
#             if search_node.emits:
#                 search_result += search_node.emits
#         return search_result
#
#
# if __name__ == "__main__":
#     data = ['he', 'she', 'his', 'hers']
#     s = "ushers"
#     ct = CreateAcAutomaton(data, "model.pkl")
#     print(ct.search(s))


#
# class TrieNode:
#     def __init__(self):
#         self.children = {}
#         self.is_end_of_word = False
#         self.fail = None
#         self.output = []
#
# class ACAutomaton:
#     def __init__(self):
#         self.root = TrieNode()
#
#     def add_pattern(self, pattern):
#         current = self.root
#         for char in pattern:
#             if char not in current.children:
#                 current.children[char] = TrieNode()
#             current = current.children[char]
#         current.is_end_of_word = True
#
#     def build(self):
#         queue = []
#         for child in self.root.children.values():
#             queue.append(child)
#             child.fail = self.root
#
#         while queue:
#             current = queue.pop(0)
#             for char, child in current.children.items():
#                 queue.append(child)
#                 fail_node = current.fail
#
#                 while fail_node and char not in fail_node.children:
#                     fail_node = fail_node.fail
#
#                 child.fail = fail_node.children[char] if fail_node else self.root
#                 child.output = child.fail.output + [child.fail] if child.fail.is_end_of_word else child.fail.output
#
#     def match(self, text):
#         current = self.root
#         matches = []
#         for i, char in enumerate(text):
#             while current and char not in current.children:
#                 current = current.fail
#             if not current:
#                 current = self.root
#                 continue
#             current = current.children[char]
#             matches.extend([j - len(pattern) + 1 for j in current.output])
#
#         return matches
#
# # 示例用法
# ac = ACAutomaton()
# ac.add_pattern("he")
# ac.add_pattern("she")
# ac.add_pattern("his")
# ac.add_pattern("hers")
# ac.build()
#
# text = "ushers"
# matches = ac.match(text)
# print(f"在文本 '{text}' 中找到的匹配位置为：{matches}")
