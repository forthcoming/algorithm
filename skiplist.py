from random import random

'''
参考 https://epaperpress.com/sortsearch/download/skiplist.pdf by William Pugh,第i层中的元素以固定概率p出现在第i+1层中
On average, each member appears in 1/(1-p) lists,The skip list contains log(1/p,n)(i.e. logarithm base 1/p of n) lists.
The expected number of steps in each linked list is at most 1/p, which can be seen by tracing the search path backwards 
from the target until reaching an member that appears in the next higher list or reaching the beginning of the current list. 
Therefore, the total expected cost of a search is 1/p*log(1/p,n) which is O(logn), when p is a constant. 
By choosing different values of p, it is possible to trade search costs against storage costs.
'''


class Node:
    def __init__(self, score: float | int, member: str):
        self.score = score
        self.member = member
        self.forward: list[Node] = []

    def __repr__(self) -> str:
        return f"Node({self.score}: {self.member})"

    @property
    def level(self) -> int:
        return len(self.forward)


class SkipList:
    def __init__(self, p: float = .25, max_level: int = 32):
        self.head = Node(0, "root")  # 头节点的score和member没有实际意义
        self.level = 0
        self.p = p
        self.max_level = max_level

    def __str__(self) -> str:
        items = list(self)
        if len(items) == 0:
            return f"SkipList(level={self.level})"

        label_size = max((len(str(item)) for item in items), default=4)
        label_size = max(label_size, 4) + 4

        node = self.head
        lines = []

        forwards = node.forward.copy()
        lines.append(f"[{node.member}]".ljust(label_size, "-") + "* " * len(forwards))
        lines.append(" " * label_size + "| " * len(forwards))

        while len(node.forward) != 0:
            node = node.forward[0]

            lines.append(
                f"[{node.score}]".ljust(label_size, "-")
                + " ".join(str(n.score) if n.score == node.score else "|" for n in forwards)
            )
            lines.append(" " * label_size + "| " * len(forwards))
            forwards[: node.level] = node.forward

        lines.append("None".ljust(label_size) + "* " * len(forwards))
        return f"SkipList(level={self.level})\n" + "\n".join(lines)

    def __iter__(self):
        node = self.head
        while len(node.forward) != 0:
            yield node.forward[0].score
            node = node.forward[0]

    def random_level(self) -> int:  # [1, self.max_level]
        level = 1
        while random() < self.p and level < self.max_level:
            level += 1

        return level

    def _locate_node(self, score) -> tuple[Node | None, list[Node]]:
        """
        :return: Tuple with searched node (or None if given score is not present)
                 and list of nodes that refer (if score is present) of should refer to given node.
        """
        # Nodes with refer or should refer to output node
        update_vector = []

        node = self.head

        for i in reversed(range(self.level)):
            # i < node.level - When node level is lesser than `i` decrement `i`.
            # node.forward[i].score < score - Jumping to node with score value higher or equal
            #                                 to searched score would result in skipping searched score.
            while i < node.level and node.forward[i].score < score:
                node = node.forward[i]
            # Each leftmost node (relative to searched node) will potentially have to
            # be updated.
            update_vector.append(node)

        update_vector.reverse()  # Note that we were inserting values in reverse order.

        # len(node.forward) != 0 - 如果当前节点不包含 any further references then searched score is not present.
        # node.forward[0].score == score - Next node score should be equal to search score if score is present.
        if len(node.forward) != 0 and node.forward[0].score == score:
            return node.forward[0], update_vector
        else:
            return None, update_vector

    def delete(self, score):
        node, update_vector = self._locate_node(score)

        if node is not None:
            for i, update_node in enumerate(update_vector):
                # Remove or replace all references to removed node.
                if update_node.level > i and update_node.forward[i].score == score:
                    if node.level > i:
                        update_node.forward[i] = node.forward[i]
                    else:
                        update_node.forward = update_node.forward[:i]

    def insert(self, score, member):
        node, update_vector = self._locate_node(score)
        if node is not None:
            node.member = member
        else:
            level = self.random_level()

            if level > self.level:
                # After level increase we have to add additional nodes to head.
                for i in range(self.level - 1, level):
                    update_vector.append(self.head)
                self.level = level

            new_node = Node(score, member)

            for i, update_node in enumerate(update_vector[:level]):
                # Change references to pass through new node.
                if update_node.level > i:
                    new_node.forward.append(update_node.forward[i])

                if update_node.level < i + 1:
                    update_node.forward.append(new_node)
                else:
                    update_node.forward[i] = new_node

    def find(self, score):
        node, _ = self._locate_node(score)
        if node is not None:
            return node.member


if __name__ == "__main__":
    skip_list = SkipList()
    skip_list.insert(2, "one")
    skip_list.insert(4, "two")
    skip_list.insert(6, "two")
    skip_list.insert(4, "three")
    skip_list.insert(8, "two")
    skip_list.insert(9, "two")
    skip_list.insert(1, "four")
    skip_list.insert(3, "five")
    print(list(skip_list))
    print(skip_list.find(4))  # three
    print(skip_list)
    skip_list.delete(4)
