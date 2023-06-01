from bisect import bisect_left, insort_right
from hashlib import md5
from struct import unpack


# Implements consistent hashing that can be used when the number of server nodes can increase or decrease.
class HashRing:
    def __init__(self, v_node_num=1):
        self.mapping = {}
        self.rings = []  # sorted list
        self.storage = {}
        self.v_node_num = v_node_num  # every node expand v_node_num times
        self.length = 0

    @staticmethod
    def _hash(value):
        k = md5(bytes(value, 'utf8')).digest()
        return unpack("<I", k[:4])[0]

    def add_items(self, items):
        if self.length:
            for item in items:
                hash_value = HashRing._hash(item)
                pos = bisect_left(self.rings, hash_value) % self.length  # 注意这里跟bisect_right的区别
                self.storage[self.rings[pos]].append(item)

    def add_node(self, node):
        for index in range(0, self.v_node_num):
            hash_value = HashRing._hash('{}#{}'.format(node, index))
            if hash_value not in self.mapping:  # 小概率事件,但也要避免,保证环里面的hash_value唯一
                self.mapping[hash_value] = node
                self.storage[hash_value] = []
                if self.length:
                    pos = bisect_left(self.rings, hash_value) % self.length
                    next_hash_value = self.rings[pos]
                    items = self.storage[next_hash_value]
                    self.storage[next_hash_value] = []
                    insort_right(self.rings, hash_value)
                    self.add_items(items)
                else:
                    self.rings.append(hash_value)
                self.length += 1

    def remove_node(self, node):
        pass


if __name__ == '__main__':
    ring = HashRing(2)
    for node in range(4):
        ring.add_node(node)
    ring.add_items(map(str, range(100)))
    print(ring.mapping)
    print(ring.rings)
    print(ring.length)
    print(ring.storage)
