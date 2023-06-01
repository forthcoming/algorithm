from bisect import bisect_left, insort_right
from hashlib import md5
from struct import unpack


# Implements consistent hashing that can be used when the number of server nodes can increase or decrease.
class HashRing:
    def __init__(self, v_node_num=1):
        self.ring = []  # sorted list,只存储节点哈希值
        self.storage = {}  # 节点哈希值与存储元素的映射关系
        self.v_node_num = v_node_num  # every node expand v_node_num times
        self.node_num = 0  # 总的节点个数

    @staticmethod
    def _hash(value):
        k = md5(bytes(value, 'utf8')).digest()
        return unpack("<I", k[:4])[0]  # unsigned int

    def add_node(self, node):
        for index in range(self.v_node_num):
            hash_value = HashRing._hash('{}#{}'.format(node, index))
            if hash_value not in self.storage:  # 小概率事件,但也要避免,保证环里面的hash_value唯一
                self.storage[hash_value] = []
                if self.node_num:
                    pos = bisect_left(self.ring, hash_value) % self.node_num
                    next_hash_value = self.ring[pos]
                    items = self.storage[next_hash_value]
                    self.storage[next_hash_value] = []
                    insort_right(self.ring, hash_value)
                    self.add_items(items)
                else:
                    self.ring.append(hash_value)
                self.node_num += 1

    def add_items(self, items):
        if self.node_num:
            for item in items:
                hash_value = HashRing._hash(item)
                pos = bisect_left(self.ring, hash_value) % self.node_num  # 注意这里跟bisect_right的区别
                self.storage[self.ring[pos]].append(item)

    def remove_node(self, node):
        pass


if __name__ == '__main__':
    ring = HashRing(2)
    for node in range(4):
        ring.add_node(node)
    ring.add_items(map(str, range(100)))
    print(ring.ring)
    print(ring.node_num)
    print(ring.storage)
