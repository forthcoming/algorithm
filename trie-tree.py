class Node:
    def __init__(self, data, flag, left=None, sibling=None):
        self.data = data
        self.flag = flag  # flag可以记录访问次数
        self.left = left
        self.sibling = sibling


class TrieTree:  # 孩子-兄弟链表存储
    def __init__(self):
        self.__root = Node('#', False)

    def search(self, word):
        root = self.__root
        for index, alpha in enumerate(word):
            self._hot = root  # 指向失败层的父节点
            self._index = index  # 指向匹配失败的位置
            root = root.left
            while root:
                if root.data == alpha:
                    break
                root = root.sibling
            else:
                return False
        return root.flag

    def insert(self, word):
        if self.search(word):
            return False
        siblings = self._hot.left
        while siblings:
            if siblings.data == word[self._index]:
                siblings.flag = True
                break
            siblings = siblings.sibling
        else:  # 没有找到
            node = Node(word[self._index], False, sibling=self._hot.left)
            self._hot.left = node
            for alpha in word[self._index + 1:]:
                son = Node(alpha, False)
                node.left = son
                node = son
            node.flag = True
        return True

    def delete(self, word):  # 直接将单词的标志位置为False即可
        if self.search(word):
            siblings = self._hot.left
            while siblings:
                if siblings.data == word[self._index]:
                    siblings.flag = False
                    break
                siblings = siblings.sibling
            return True
        return False

    def show(self):
        root = self.__root.left
        stack = []

        def _traverse(root):
            while root:
                stack.append(root.data)
                if root.left:
                    _traverse(root.left)
                if root.flag:
                    print(''.join(stack))
                stack.pop()
                root = root.sibling

        _traverse(root)


if __name__ == '__main__':
    '''
                 #                                                                     #
              /     \                                                                 /
             i       j                                                               i
             |     / | \       [in,init,ink,jack,jar,john,just,june,junk]         /     \
             n    a  o  u                                                        n       j
            / \  / \ | / \                                                      /       /
           i   k c r h s  n                                                    i       a
           |     |   | | / \                                                  / \     / \
           t     k   n t e k                                                 t   k   c   o
                                                                                    / \ / \
                                                                                    k r h u
                                                                                       / /
                                                                                      n s
                                                                                       / \
                                                                                      t   n
                                                                                         /
                                                                                        e
                                                                                         \
                                                                                          k
    '''
    trie = TrieTree()
    for word in ['in', 'init', 'ink', 'jack', 'jar', 'john', 'just', 'june', 'junk']:
        trie.insert(word)
    print(trie.delete('john'))
    trie.show()
