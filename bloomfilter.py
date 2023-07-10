import copy
import math
import random

import redis


class BloomFilter:
    rds = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=10)
    pipeline = rds.pipeline()

    def __init__(self, name, capacity=10000000, error_rate=.01):
        """
        假定m,n固定,当k=ln2*m/n时,错误率p达到最小,此时p=(.5**ln2)**(m/n)=.6185**(m/n)
        若k=m/n,此时p0=(1-1/e)**(m/n)=.6321**(m/n),很显然当k取值不当时,在相同p,n情况下,m必须更大,显然会造成空间浪费
        k可进一步化简,k=-log(2,p),由此可见k仅与错误率p有关

        capacity:   预先估计要去重的元素数量
        error_rate: 错误率
        m:          所需要的比特位个数
        """
        self.__name = f'bloomfilter:{name}'
        self.n = capacity
        self.p = error_rate
        # log2(*args, **kwargs) return the base 2 lo logarithm of x
        self.m = math.ceil(-capacity * math.log2(math.e) * math.log2(error_rate))
        self.k = math.ceil(-math.log(error_rate, 2))  # log(x, [base=math.e])
        self.rds.delete(self.name)
        self.hash_functions = self.generate_hash_functions()
        print(f'至少需要{self.m}个bit,{self.k}次哈希,内存占用{self.m / 2 ** 23}M')  # m/8/1024/1024

    @property
    def name(self):  # name对外只读
        return self.__name

    @staticmethod
    def BKDRHash(key, radix=31):
        # radix 31 131 1313 13131 131313 etc.
        key = bytearray(key.encode())
        value = 0
        for i in key:
            value = value * radix + i
        return value

    @staticmethod
    def DJBHash(key):
        value = 5381
        for i in key:
            value = ((value << 5) + value) + ord(i)
        return value

    @staticmethod
    def JSHash(key):
        value = 1315423911
        for i in key:
            value ^= ((value << 5) + ord(i) + (value >> 2))
        return value

    def generate_hash_functions(self):
        random.seed(0)  # 重要,保证后面的union和intersection能正确运算
        functions = [__class__.BKDRHash, __class__.DJBHash, __class__.JSHash]
        hash_functions = []
        for idx in range(self.k):
            zoomin = random.randint(1, self.m)
            offset = random.randint(1, self.m)
            function = random.choice(functions)
            # 也可以通过yield lambda key:(zoomin * hash_function(str(key)) + offset) % self.m消除延时绑定带来的副作用,
            # 但每次会重新计算zoomin/offset/function,效率低
            hash_functions.append(lambda key, z=zoomin, o=offset, f=function: (z * f(str(key)) + o) % self.m)
        return hash_functions

    def add(self, keys):
        for key in keys:
            for hash_function in self.hash_functions:
                self.pipeline.setbit(self.name, hash_function(key), 1)
        self.pipeline.execute()

    def check(self, key):
        for hash_function in self.hash_functions:
            self.pipeline.getbit(self.name, hash_function(key))
        for result in self.pipeline.execute():
            if not result:
                return False
        return True

    def count(self):  # Approximating the number of items in a Bloom filter
        """
        EstimatedCount = -(NumBits * ln(1 – BitsOn / NumBits)) / NumHashes
        EstimatedCount = math.log(1-BitsOn/NumBits , 1-1/NumBits) / NumHashes       Ⅱ
        Ⅱ推导过程如下: EstimatedCount个元素插入后某个bit位仍然为0的概率 * NumBits = NumBits - BitsOn
        思想是先求出某个量的数学期望,再让他等于实际观测值,从而解出某些未知量
        """
        X = self.rds.bitcount(self.name)
        return math.ceil(-self.m / self.k * math.log(1 - X / self.m))

    # Calculates the union of the two underlying bitarrays and returns a new bloom filter object
    def __or__(self, other):
        if self.n != other.n or self.p != other.p:
            raise ValueError("Unioning filters requires both filters to have both the same capacity and error rate")
        bf = copy.copy(self)
        bf.__name = 'bloomfilter:or'
        self.rds.bitop('or', bf.name, self.name, other.name)
        return bf

    # Calculates the intersection of the two underlying bitarrays and returns a new bloom filter object
    def __and__(self, other):
        # Count(Intersection(A,B)) = (Count(A) + Count(B)) – Count(Union(A,B))
        if self.n != other.n or self.p != other.p:
            raise ValueError("Intersecting filters requires both filters to have equal capacity and error rate")
        bf = copy.copy(self)
        bf.__name = 'bloomfilter:and'
        self.rds.bitop('and', bf.name, self.name, other.name)
        return bf

    def jaccard_index(self, other):
        # Jaccard Index = Count(Intersection(A,B)) / Count(Union(A,B))
        if self.n != other.n or self.p != other.p:
            raise ValueError("jaccard index requires both filters to have equal capacity and error rate")
        count_or = (self | other).count()
        if count_or:
            return (self.count() + other.count()) / count_or - 1


if __name__ == '__main__':
    A = BloomFilter('A', 2000000)
    A.add(range(20000))
    B = BloomFilter('B', 2000000)
    B.add(range(15000, 35000))

    A_or_B = A | B
    A_and_B = A & B
    A_cnt = A.count()
    B_cnt = B.count()
    A_or_B_cnt = A_or_B.count()
    A_and_B_cnt0 = A_cnt + B_cnt - A_or_B_cnt
    A_and_B_cnt1 = A_and_B.count()

    print(A.check(36))  # True
    print(A.check('akatsuki'))  # False
    print(A_cnt)  # 19994
    print(B_cnt)  # 19987
    print(A_or_B_cnt)  # 34999
    print(A_and_B_cnt0)  # 4982
    print(A_and_B_cnt1)  # 5064
    print(A.jaccard_index(B))  # 0.14234692419783412
