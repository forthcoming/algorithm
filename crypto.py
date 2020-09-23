import random,binascii,hashlib

'''
欧拉函数:
比n小但与n互素的正整数个数φ(n)称为n的欧拉函数
对任一素数p,有φ(n)＝p-1,对于两个不同的素数p和q则φ(pq)=(p-1)(q-1)=φ(p)*φ(q)

模运算：
(a + b) % p = (a % p + b % p) % p
(a - b) % p = (a % p - b % p) % p
(a * b) % p = (a % p * b % p) % p   
a ^ b % p = ((a % p)^b) % p

辗转相除法: 
二个整數a、b,必存在整數x、y使得ax + by = gcd(a,b),可由矩阵推导证明
对a,b进行辗转相除,可得它们的最大公约数,然后收集辗转相除法中产生的式子,倒回去可以得到ax+by=gcd(a,b)的整数解

模反元素(前提是互素):
如果正整数a和b互素,那么一定可以找到整数d,使得ad-1被b整除(可用辗转相除法矩阵乘法证),如果d是a的模反元素,则d+kb都是a的模反元素,比如3和11互质,那么3的模反元素就是4
如果gcd(a, b) = 1,則稱a和b互素(除了1以外没有其他公因子),a和b是否互素和它们是否素数无关,最小公倍数(a,b)=a*b/gcd(a,b)
ax+by=1,则a,b互素
(ax+by)%b=1%b 
ax%b=1 
即求出a模b的逆元为x

欧拉定理: 
如果两个正整数a和n互质,n的欧拉函数φ(n),则a^φ(n) % n = 1, 比如3和7互质,而7的欧拉函数φ(7)等于6,所以3的6次方729减去1,可以被7整除(728/7=104)

RSA步骤:
1. 随机选择两个不相等的大质数p和q,比如61和53; 不相等大素数除了保证因式分解困难外,还确保了对任意0<=m<n都满足m^(ed)%n = m
2. 计算n = p*q = 61×53 = 3233,n的长度就是密钥长度,3233写成二进制是110010100001,一共有12位,所以这个密钥就是12位,实际应用中,RSA密钥一般是1024位或2048位
3. 计算φ(n) = (p-1)(q-1) = 60*52 = 3120
4. 随机选择一个整数e,条件是1< e < φ(n)且e与φ(n)互质,比如e = 17
5. 计算e对于φ(n)的模反元素d,即满足ed%φ(n) = 1,这个方程可以用"扩展欧几里得算法"求解,如d=2753
6. 将n和e封装成公钥,n和d封装成私钥,一旦d泄漏就等于私钥泄漏
7. 加密名文m=65需要用到公钥(n,e),密文c=m^e%n=65^17%3233=2790;注意m必须是整数(字符串可以取ascii值或unicode值)且必须小于n(由加密解密证明可以看到m必须小于n)
8. 解密密文c=2790需要用到私钥(n,d),明文m=c^d%n=2790^2753%3233=65

说明:
公钥和私钥完全对等,所以也可以用私钥加密,公钥解密来做签名

RSA安全性:
n,e已知,要想知道d必须知道φ(n),就必须对n做因式分解,但大整数分解质因数很困难
c = m^e%n,但由密文c推导明文m很难

RSA加密解密证明:
即证明m = c^d%n,又因为m^e%n = c,转化为证m = m^(ed)%n
又因为ed = hφ(n)+1,转化为证m = m^(hφ(n)+1)%n   ToBeDone...
'''

class Base:

    @staticmethod
    def is_probable_prime(n, trials = 10): # Miller-Rabin检测,error_rate=.25**trials
        assert n > 1
        if n == 2: # 2是素数
            return True
        if not n&1: # 排除偶数
            return False
        s = 0
        d = n - 1
        while not d&1:  # 把n-1写成(2^s)*d的形式
            s+=1
            d>>=1
         
        for i in range(trials):
            a = random.randrange(2, n)  # 每次的底a是不一样的,只要有一次未通过测试,则判定为合数
            if __class__.power(a, d, n) != 1: # 相当于(a^d)%n
                for r in range(s):
                    if __class__.power(a, 2 ** r * d, n) == n - 1: #相当于(a^((2^i)*d))%n
                        break
                else:
                    return False  # 以上条件都满足时,n一定是合数
        return True

    @staticmethod
    def power(a,b,r):  # a**b%r or pow(a,b,r) 
        res=1
        while b:
            if b&1:
                res=res*a%r  # 防止数字过大导致越界
            b>>=1    # 隐式减去了1
            a=a*a%r  # 防止a增大
        return res

    @staticmethod
    def power_v1(x,y):  # y可以是任意整数
        # 分治法降低power时间复杂度到logn,效率 x**y = pow > power_v1
        result = 1
        if y<0:
            x=1/x
            y=-y
        while y:
            if y&1:
                result *= x
            x *= x
            y >>= 1
        return result

    @staticmethod
    def exgcd_iter(a, b): 
        x = 0
        y = 1
        lx = 1
        ly = 0
        _b = b
        while b != 0:
            q = a // b
            a, b = b, a % b
            x, lx = lx - q * x, x
            y, ly = ly - q * y, y
        if lx < 0:
            lx += _b
        return lx, a
    
    @staticmethod
    def exgcd(a,b):  # 只有当a,b互素时算出的d才有实际意义
        '''
        对于a' = b, b' = a%b = a - a / b * b而言,我们求得d, y使得a' d+b' y=gcd(a', b') 
        ===>
        bd + (a - a/b *b)y = gcd(a' , b') = gcd(a, b) , 注意到这里的/是C语言中的除法 
        ===>
        ay + b(d- a/b *y) = gcd(a, b)
        因此对于a和b而言,他们的相对应的p,q分别是y和(d-a/b*y)
        '''
        def _exgcd(a, b):
            if b:
                d , y , common_divisor = _exgcd( b , a % b )
                d , y = y, d - (a // b) * y
            else:
                '''
                当b=0时gcd(a,b) = a ; ad + 0y = a
                所以d=1, y可以是任意数,但一般选0,这样会使所求的模反元素最小, common_divisor=a
                '''
                d, y, common_divisor = 1, 0, a
            return d,y,common_divisor
        d,y,common_divisor = _exgcd(a,b)
        while d<0: # 如果d是a的模反元素(ad%b=1),则d+kb也是a的模反元素,RSA算法要求d是正数
            d+=b
        return d,common_divisor

    @staticmethod
    def exgcd_mat(a, b):  # 矩阵版(numpy缺点是处理大整数溢出)
        '''
        a=q0*b+r1
        b=q1*r1+r2
        r1=q2*r2+r3
        ......
        rn-1=qn*rn+0  
        a    q0 1   q1 1          qn 1   rn        rn
          =       *      ...... *      *     = M * 
        b    1  0   1  0          1  0   0          0
        此处的rn即为最大公约数
        '''
        import numpy as np
        _b = b
        M=np.eye(2,dtype=np.int64)
        while b:
            M=M@np.array([[a//b,1],[1,0]])  # 注意不能用*
            a,b=b,a%b
        D=M[0][0]*M[1][1]-M[1][0]*M[0][1]   # 计算行列式(值是1或者-1,取决于循环的次数)
        d = M[1][1] // D                    # M的逆矩阵M' = M* / D, M[1][1]对应M*[0][0]
        while d<0: 
            d += _b
        return d,a

    @staticmethod
    def factorization(n): # 因式分解
        factor=2
        while n!=1:
            idx=0
            while not n%factor:
                idx+=1
                n /= factor
            if idx:
                print(factor,idx)
            factor+=1

class RSA(Base):
    def __init__(self):
        # P,Q在初始化后应当被销毁,防止外泄
        P = self.generate_prime()
        Q = self.generate_prime()
        while P == Q:
            Q = self.generate_prime()
        phi=(P-1) * (Q-1)
        self.module = P * Q  # 公钥
        self.e = random.randrange(3,phi,2)  # 公钥,跟phi互质的任意数,这里必须是奇数
        self.d,common_divisor=self.exgcd(self.e,phi) # 私钥
        while common_divisor!=1:
            self.e+=2
            self.d,common_divisor=self.exgcd(self.e,phi)
    
    @staticmethod
    def generate_prime():
        prime=random.randrange((1<<199)+1,1<<300,2)
        while not __class__.is_probable_prime(prime):
            prime+=2
        return prime

    def encryption(self,message): 
        message=int(binascii.hexlify(bytes(message,encoding='utf8')),16)
        assert(0<=message<self.module)
        return self.power(message,self.e,self.module)

    def decryption(self,cipher):
        cipher=self.power(cipher,self.d,self.module)
        # return binascii.unhexlify(bytes(hex(cipher),encoding='utf8')[2:])
        res=[]
        while cipher:
            res.append(chr(cipher&255))
            cipher>>=8
        return ''.join(res[::-1])

class DSA(Base):  # DSA和RSA不同之处在于它不能用作加密和解密,也不能进行密钥交换,只用于签名,它比RSA要快很多
    def __init__(self):
        self.q=random.randrange((1<<159)+1,1<<160,2) # 公钥,160bit位奇数里面挑选
        while not self.is_probable_prime(self.q):
            self.q+=2

        factor=1<<300 
        self.p=factor*self.q+1  # 公钥
        while not self.is_probable_prime(self.p):
            factor+=1
            self.p=factor*self.q+1

        self.g=pow(random.randrange(2,self.p-1),factor,self.p)  # 公钥
        self.x=random.randrange(1,self.q)  # 私钥
        self.y=pow(self.g,self.x,self.p)  # 公钥
   
    def sha(self,message):
        return int(hashlib.sha256(message.encode('utf8')).hexdigest(),16)

    def sign(self,message):
        k=random.randrange(1,self.q)
        r=pow(self.g,k,self.p)%self.q
        Hm=self.sha(message)
        s=(Hm+self.x*r)*self.exgcditer(k,self.q)[0]
        return r,s

    def check(self,message,r,s):
        w=self.exgcditer(s,self.q)[0]
        Hm=self.sha(message)
        u1=Hm*w%self.q
        u2=r*w%self.q
        v=pow(self.g,u1,self.p)*pow(self.y,u2,self.p)%self.p%self.q
        return v==r

if __name__ == "__main__":
    rsa=RSA() 
    message='akatsuki'
    print(rsa.decryption(rsa.encryption(message)))

    dsa=DSA()
    message='avatar'
    r,s=dsa.sign(message)
    print(dsa.check(message,r,s))  # True
    
