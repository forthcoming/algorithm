import random,math,redis

conn = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=10)  # socket_connect_timeout设置连接超时时间

class BloomFilter:
    def __init__(self, conn, name, capacity=10000000, error_rate=.01):
        '''
        假定m,n固定,当k=ln2*m/n时,错误率p达到最小,此时p=(.5**ln2)**(m/n)=.6185**(m/n)
        若k=m/n,此时p0=(1-1/e)**(m/n)=.6321**(m/n),很显然当k取值不当时,在相同p,n情况下,m必须更大,显然会造成空间浪费
        k可进一步化简,k=-log(2,p),由此可见k仅与错误率p有关
        
        capacity:   预先估计要去重的元素数量
        error_rate: 错误率
        m:          所需要的比特位个数
        '''   
        self.conn = conn
        self.name = f'bloomfilter:{name}'
        self.n = capacity
        self.p = error_rate
        self.m = math.ceil(-capacity*math.log2(math.e)*math.log2(error_rate)) # log2(*args, **kwargs) return the base 2 lo garithm of x
        self.k = math.ceil(-math.log(error_rate,2))                           # log(x, [base=math.e])
        self.conn.delete(self.name)
        self.functions = [__class__.BKDRHash,__class__.DJBHash,__class__.JSHash]
        print(f'至少需要{self.m}个bit,{self.k}次哈希,内存占用{self.m>>23}M')  # m/8/1024/1024

    @staticmethod
    def BKDRHash(key,radix=31):
        # radix 31 131 1313 13131 131313 etc.
        key = bytearray(key.encode())   
        value=0
        for i in key:
            value=value*radix+i
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

    def hash_functions(self):
        random.seed(0)  # 重要,保证后面的union和intersection能正确运算
        for idx in range(self.k):
            zoomin = random.randint(1,self.m)
            offset = random.randint(1,self.m)
            hash_function = random.choice(self.functions)
            yield lambda key:(zoomin * hash_function(str(key)) + offset) % self.m # yield用来消除延时绑定带来的副作用,同时节省了内存

    def add(self,key):
        for hash_function in self.hash_functions():
            self.conn.setbit(self.name,hash_function(key),1)

    def check(self,key):
        for hash_function in self.hash_functions():
            if not self.conn.getbit(self.name,hash_function(key)):
                return False
        return True

    def count(self): # Approximating the number of items in a Bloom filter
        # EstimatedCount = -(NumBits * ln(1 – BitsOn / NumBits)) / NumHashes
        X = self.conn.bitcount(self.name)
        return math.ceil(-self.m/self.k*math.log(1-X/self.m)) # 注意不要使用其他等价形式计算公式如math.ceil(-self.n*math.log2(1-X/self.m))

    def __or__(self,other):  # Calculates the union of the two underlying bitarrays and returns a new bloom filter object
        if self.n != other.n or self.p != other.p:
            raise ValueError("Unioning filters requires both filters to have both the same capacity and error rate")
        bf = BloomFilter(self.conn,'bloomfilter:or', self.n, self.p)
        bf.conn.bitop('or',bf.name,self.name,other.name)
        return bf

    def __and__(self,other):  # Calculates the intersection of the two underlying bitarrays and returns a new bloom filter object
        # Count(Intersection(A,B)) = (Count(A) + Count(B)) – Count(Union(A,B))
        if self.n != other.n or self.p != other.p:
            raise ValueError("Intersecting filters requires both filters to have equal capacity and error rate")
        bf = BloomFilter(self.conn, 'bloomfilter:and', self.n, self.p)
        bf.conn.bitop('and',bf.name,self.name,other.name)
        return bf

    def jaccard_index(self,other):
        # Jaccard Index = Count(Intersection(A,B)) / Count(Union(A,B))
        if self.n != other.n or self.p != other.p:
            raise ValueError("jaccard index requires both filters to have equal capacity and error rate")
        a_or_b = self|other
        return (self.count()+other.count())/a_or_b.count() - 1 

if __name__=='__main__':
    A = BloomFilter(conn,'A',10000)
    B = BloomFilter(conn,'B',10000)
    for i in range(200):
        A.add(i)
    for i in range(150,350):
        B.add(i)
    A_or_B = A|B
    A_and_B = A&B

    A.check('akatsuki') # False
    A.check(36)         # True

    A_cnt = A.count()   
    B_cnt = B.count()
    A_or_B_cnt = A_or_B.count()
    A_and_B_cnt0 = A_cnt+B_cnt-A_or_B_cnt
    A_and_B_cnt1 = A_and_B.count()
    print(A_cnt)              # 201
    print(B_cnt)              # 201
    print(A_or_B_cnt)         # 351
    print(A_and_B_cnt0)       # 51
    print(A_and_B_cnt1)       # 51
    print(A.jaccard_index(B)) # 0.14613180515759305


