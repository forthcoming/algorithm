import random,math,redis

class Hash:
    def __init__(self,m):
        self.m=m
        self.zoomin=random.randint(1,m)
        self.offset=random.randint(1,m)
        self.fun=random.choice(['BKDRHash','DJBHash','JSHash'])

    def BKDRHash(self,key,radix=31):
        # radix 31 131 1313 13131 131313 etc.
        key = bytearray(key.encode())   
        hash=0
        for i in key:
            hash=hash*radix+i
        return hash

    def DJBHash(self,key):
        hash = 5381
        for i in key:
           hash = ((hash << 5) + hash) + ord(i)
        return hash

    def JSHash(self,key):
        hash = 1315423911
        for i in key:
            hash ^= ((hash << 5) + ord(i) + (hash >> 2))
        return hash
    
    def hash(self,key):
        return (self.zoomin *getattr(self,self.fun)(key) + self.offset) % self.m

class BloomFilter:
    def __init__(self, conn, name, capacity=1000000000, error_rate=.001):
        '''
        假定m,n固定,当k=ln2*m/n时,错误率p达到最小,此时p=(.5**ln2)**(m/n)=.6185**(m/n)
        若k=m/n,此时p0=(1-1/e)**(m/n)=.6321**(m/n),很显然当k取值不当时,在相同p,n情况下,m必须更大,显然会造成空间浪费
        k可进一步化简,k=-log(2,p),由此可见k仅与错误率p有关
        
        capacity:   预先估计要去重的元素数量
        error_rate: 错误率
        m:          所需要的比特位个数
        '''   
        m = math.ceil(-capacity*math.log2(math.e)*math.log2(error_rate)) # log2(*args, **kwargs) return the base 2 lo garithm of x
        k = math.ceil(-math.log(error_rate,2))                           # log(x, [base=math.e])
        self.name = name
        self.conn = conn
        self.hashFunc=[Hash(m) for i in range(k)]
        print(f'至少需要{m}个bit,{k}次哈希,内存占用{m>>23}M')  # m/8/1024/1024

    def add(self,key):
        for _ in self.hashFunc:
            self.conn.setbit(self.name,_.hash(key),1)

    def check(self,key):
        for _ in self.hashFunc:
            if not self.conn.getbit(self.name,_.hash(key)):
                return False
        return True

if __name__=='__main__':
    conn = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=10)  # socket_connect_timeout设置连接超时时间
    bf=BloomFilter(conn,'bf',20)
    for key in ['avatar','akatsuki','avatar','10086','wanted','hunter','fork',]:
        bf.add(key)
    print(bf.check('avatar'))
    print(bf.check('apple'))
    print(bf.check('10086'))
