class Node:
    def __init__(self, data, flag=False, left=None, sibling=None):
        self.data = data
        self.flag = flag  # flag可以记录访问次数
        self.left = left
        self.sibling = sibling


class TrieTree:  # 孩子-兄弟链表存储
    def __init__(self):
        self._index = -1
        self._hot = None
        self.__root = Node('#')

    def search(self, word):
        """
        没找到情况:
        1. trie某一层(包括叶子结点后面的一层)没有字符与word对应
        2. 匹配完word,但flag=False即word仅为某个单词前缀
        """
        root = self.__root
        for index, alpha in enumerate(word):
            self._hot = root  # 指向遍历层的父节点
            self._index = index  # 指向匹配字符位置
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
                node.left = Node(alpha)
                node = node.left
            node.flag = True
        return True

    def delete(self, word):  # 简单将单词的标志位置为False(严格点说需要将没用的树枝都删掉)
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

        _traverse(self.__root.left)


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
    print(trie.delete('ink'))
    print(trie.search("ini"))
    trie.show()
