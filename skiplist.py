from random import random
from typing import Generic, List, Optional, Tuple, TypeVar

KT = TypeVar("KT")
VT = TypeVar("VT")


'''
Based on "Skip Lists: A Probabilistic Alternative to Balanced Trees" by William Pugh https://epaperpress.com/sortsearch/download/skiplist.pdf

where an element in layer i appears in layer i+1 with some fixed probability p(two commonly used values for p are 1/2 or 1/4).
On average, each element appears in 1/(1-p) lists,The skip list contains log(1/p,n)(i.e. logarithm base 1/p of n) lists.
The expected number of steps in each linked list is at most 1/p, which can be seen by tracing the search path backwards from the target until reaching an element that appears in the next higher list or reaching the beginning of the current list. 
Therefore, the total expected cost of a search is 1/p*log(1/p,n) which is O(logn), when p is a constant. By choosing different values of p, it is possible to trade search costs against storage costs.
'''
class Node(Generic[KT, VT]):
    def __init__(self, key: KT, value: VT):
        self.key = key
        self.value = value
        self.forward: List[Node[KT, VT]] = []

    def __repr__(self) -> str:
        """
        :return: Visual representation of Node
        >>> node = Node("Key", 2)
        >>> repr(node)
        'Node(Key: 2)'
        """

        return f"Node({self.key}: {self.value})"

    @property
    def level(self) -> int:
        """
        :return: Number of forward references
        >>> node = Node("Key", 2)
        >>> node.level
        0
        >>> node.forward.append(Node("Key2", 4))
        >>> node.level
        1
        >>> node.forward.append(Node("Key3", 6))
        >>> node.level
        2
        """

        return len(self.forward)


class SkipList(Generic[KT, VT]):
    def __init__(self, p: float = 0.5, max_level: int = 16):
        self.head = Node("root", None)
        self.level = 0
        self.p = p
        self.max_level = max_level

    def __str__(self) -> str:
        """
        :return: Visual representation of SkipList
        >>> skip_list = SkipList()
        >>> print(skip_list)
        SkipList(level=0)
        >>> skip_list.insert("Key1", "Value")
        >>> print(skip_list) # doctest: +ELLIPSIS
        SkipList(level=...
        [root]--...
        [Key1]--Key1...
        None    *...
        >>> skip_list.insert("Key2", "OtherValue")
        >>> print(skip_list) # doctest: +ELLIPSIS
        SkipList(level=...
        [root]--...
        [Key1]--Key1...
        [Key2]--Key2...
        None    *...
        """

        items = list(self)

        if len(items) == 0:
            return f"SkipList(level={self.level})"

        label_size = max((len(str(item)) for item in items), default=4)
        label_size = max(label_size, 4) + 4

        node = self.head
        lines = []

        forwards = node.forward.copy()
        lines.append(f"[{node.key}]".ljust(label_size, "-") + "* " * len(forwards))
        lines.append(" " * label_size + "| " * len(forwards))

        while len(node.forward) != 0:
            node = node.forward[0]

            lines.append(
                f"[{node.key}]".ljust(label_size, "-")
                + " ".join(str(n.key) if n.key == node.key else "|" for n in forwards)
            )
            lines.append(" " * label_size + "| " * len(forwards))
            forwards[: node.level] = node.forward

        lines.append("None".ljust(label_size) + "* " * len(forwards))
        return f"SkipList(level={self.level})\n" + "\n".join(lines)

    def __iter__(self):
        node = self.head

        while len(node.forward) != 0:
            yield node.forward[0].key
            node = node.forward[0]

    def random_level(self) -> int:
        """
        :return: Random level from [1, self.max_level] interval.
                 Higher values are less likely.
        """

        level = 1
        while random() < self.p and level < self.max_level:
            level += 1

        return level

    def _locate_node(self, key) -> Tuple[Optional[Node[KT, VT]], List[Node[KT, VT]]]:
        """
        :param key: Searched key,
        :return: Tuple with searched node (or None if given key is not present)
                 and list of nodes that refer (if key is present) of should refer to
                 given node.
        """

        # Nodes with refer or should refer to output node
        update_vector = []

        node = self.head

        for i in reversed(range(self.level)):
            # i < node.level - When node level is lesser than `i` decrement `i`.
            # node.forward[i].key < key - Jumping to node with key value higher
            #                             or equal to searched key would result
            #                             in skipping searched key.
            while i < node.level and node.forward[i].key < key:
                node = node.forward[i]
            # Each leftmost node (relative to searched node) will potentially have to
            # be updated.
            update_vector.append(node)

        update_vector.reverse()  # Note that we were inserting values in reverse order.

        # len(node.forward) != 0 - If current node doesn't contain any further
        #                          references then searched key is not present.
        # node.forward[0].key == key - Next node key should be equal to search key
        #                              if key is present.
        if len(node.forward) != 0 and node.forward[0].key == key:
            return node.forward[0], update_vector
        else:
            return None, update_vector

    def delete(self, key: KT):
        """
        :param key: Key to remove from list.
        >>> skip_list = SkipList()
        >>> skip_list.insert(2, "Two")
        >>> skip_list.insert(1, "One")
        >>> skip_list.insert(3, "Three")
        >>> list(skip_list)
        [1, 2, 3]
        >>> skip_list.delete(2)
        >>> list(skip_list)
        [1, 3]
        """

        node, update_vector = self._locate_node(key)

        if node is not None:
            for i, update_node in enumerate(update_vector):
                # Remove or replace all references to removed node.
                if update_node.level > i and update_node.forward[i].key == key:
                    if node.level > i:
                        update_node.forward[i] = node.forward[i]
                    else:
                        update_node.forward = update_node.forward[:i]

    def insert(self, key: KT, value: VT):
        """
        :param key: Key to insert.
        :param value: Value associated with given key.
        >>> skip_list = SkipList()
        >>> skip_list.insert(2, "Two")
        >>> skip_list.find(2)
        'Two'
        >>> list(skip_list)
        [2]
        """

        node, update_vector = self._locate_node(key)
        if node is not None:
            node.value = value
        else:
            level = self.random_level()

            if level > self.level:
                # After level increase we have to add additional nodes to head.
                for i in range(self.level - 1, level):
                    update_vector.append(self.head)
                self.level = level

            new_node = Node(key, value)

            for i, update_node in enumerate(update_vector[:level]):
                # Change references to pass through new node.
                if update_node.level > i:
                    new_node.forward.append(update_node.forward[i])

                if update_node.level < i + 1:
                    update_node.forward.append(new_node)
                else:
                    update_node.forward[i] = new_node

    def find(self, key: VT) -> Optional[VT]:
        """
        :param key: Search key.
        :return: Value associated with given key or None if given key is not present.
        >>> skip_list = SkipList()
        >>> skip_list.find(2)
        >>> skip_list.insert(2, "Two")
        >>> skip_list.find(2)
        'Two'
        >>> skip_list.insert(2, "Three")
        >>> skip_list.find(2)
        'Three'
        """

        node, _ = self._locate_node(key)

        if node is not None:
            return node.value

        return None

def main():
    skip_list = SkipList()
    skip_list.insert(2, "2")
    skip_list.insert(4, "4")
    skip_list.insert(6, "4")
    skip_list.insert(4, "5")
    skip_list.insert(8, "4")
    skip_list.insert(9, "4")
    skip_list.delete(4)

    print(skip_list)


if __name__ == "__main__":
    main()
    
