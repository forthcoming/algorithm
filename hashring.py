from bisect import bisect_left,insort_right
from hashlib import md5
from struct import unpack

class HashRing:    # Implements consistent hashing that can be used when the number of server nodes can increase or decrease.

    def __init__(self,v_node_num=10):
        self.mapping = {}
        self.rings = []             # sorted list
        self.storage={}
        self.v_node_num=v_node_num  # every node expand v_node_num times

    def _hash(self,value):
        k = md5(bytes(value,'utf8')).digest()
        return unpack("<I", k[:4])[0]

    def add_items(self,items):
        length=len(self.rings)
        if length:
            for item in items:
                hash_value = self._hash(item)
                pos = bisect_left(self.rings, hash_value) % length  # 注意这里跟bisect_right的区别
                self.storage[self.mapping[self.rings[pos]]].append(item)

    def add_node(self,node):
        if node not in self.storage:
            self.storage[node]=[]
            for index in range(0, self.v_node_num):
                hash_value = self._hash('{}#{}'.format(node,index))
                if hash_value not in self.mapping:   # 小概率事件,但也要避免
                    self.mapping[hash_value] = node 
                    length=len(self.rings)  
                    if length:
                        pos=bisect_left(self.rings, hash_value) % length
                        _node=self.mapping[self.rings[pos]]
                        insort_right(self.rings,hash_value)
                        if _node!=node:
                            items=self.storage[_node]
                            self.storage[_node]=[]
                            self.add_items(items)
                    else:
                        self.rings.append(hash_value) 

    def remove_node(self,node):
        pass

if __name__=='__main__':
    ring = HashRing()
    for node in range(50):
        ring.add_node(node)
    ring.add_items([str(i) for i in range(1000000)])
