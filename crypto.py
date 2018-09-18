import random,binascii,hashlib

'''
比n小但与n互素的正整数个数φ(n)称为n的欧拉函数
对任一素数p,有φ(n)＝p-1,对于两个不同的素数p和q则φ(pq)=(p-1)(q-1)=φ(p)*φ(q)
二个整數a、b,必存在整數x、y使得ax + by = gcd(a,b),可由矩阵推导证明
对a,b进行辗转相除,可得它们的最大公约数,然后收集辗转相除法中产生的式子,倒回去可以得到ax+by=gcd(a,b)的整数解,可以用来计算模反元素(也叫模逆元)
如果gcd(a, b) = 1，則稱a和b互素(除了1以外没有其他公因子),a和b是否互素和它们是否素数无关
ax+by=1,则a,b互素
(ax+by)%b=1%b 
ax%b=1 
即求出a模b的逆元为x
密钥不能出现负数,明文message满足0<=message<module
RSA可靠性:1. 大整数因数分解困难; 2. message**e%module=secret但由secret推导message很难
RSA生成的两个大素数除了保证因数分解困难外,还确保了对任意0<=message<module都满足message**(ed)%module=message(分2种情况讨论,需用到欧拉定理)
模运算：
(a + b) % p = (a % p + b % p) % p
(a - b) % p = (a % p - b % p) % p
(a * b) % p = (a % p * b % p) % p   
a ^ b % p = ((a % p)^b) % p
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
            b>>=1 # 隐式减去了1
            a=a*a%r
        return res

    @staticmethod
    def exgcd(a,b):
        def _exgcd( a , b ):   # 整数a對模数b之模反元素存在的充分必要條件是a和b互質
            if b:
                x , y , remainder = _exgcd( b , a % b )
                x , y = y, ( x - (a // b) * y )
                return x, y, remainder
            return 1, 0, a
        x,y,remainder=_exgcd(a,b)
        while x<0:  # 此时求的是最小的模反元素,还需要将它转换成正数
            x+=b
        return x,remainder
        '''
        和gcd递归实现相比,发现多了下面的x,y赋值过程,可以这样思考: 对于a' =b , b' =a%b 而言，我们求得x, y使得a' x+b' y=gcd(a', b') 由于b' = a % b = a - a / b * b 那么可以得到
        a' x + b' y = gcd(a' , b')
        ===>
        bx + (a - a/b *b)y = gcd(a' , b') = Gcd(a, b)  //注意到这里的/是C语言中的出发
        ===>
        ay + b(x- a/b *y) = gcd(a, b)
        因此对于a和b而言，他们的相对应的p，q分别是y和(x-a/b*y)
        '''

    @staticmethod
    def exgcd_slow(a , b):
        import numpy as np
        M=np.eye(2,dtype=np.int64)
        while b:
            M=M@np.array([[a//b,1],[1,0]])  # 注意不能用*
            a,b=b,a%b
        D=M[0][0]*M[1][1]-M[1][0]*M[0][1]  # 计算行列式
        return M[1][1]//D,-M[0][1]//D  # 两个整数分别是s = (−1)**(N+1)*m22、t = (−1)**N*m12
        '''
        a=q0*b+r1
        b=q1*r1+r2
        r1=q2*r2+r3
        ......
        rn-1=qn*rn+0
        a    q0 1   q1 1          qn 1   rn
          =       *      ...... *      * 
        b    1  0   1  0          1  0   0
        '''

    @staticmethod
    def gcd(x,y):
        while y:
            x,y=y,x%y
        return x
    # 递归版  
    # if y:
    #     return gcd(y,x%y)
    # return x

    @staticmethod
    def factorization(n): # 因式分解
        factor=2
        while n!=1:
            idx=0
            while not n%factor:
                idx+=1
                n/= factor
            if idx:
                print(factor,idx)
            factor+=1

class RSA(Base):
    def __init__(self):
        # P,Q在初始化后应当被销毁,防止外泄
        P = self.generate_prime()
        Q = self.generate_prime()
        phi=(P-1) * (Q-1)
        self.module = P * Q  # 公钥
        self.e = random.randrange(3,phi,2)  # 公钥,跟phi互质的任意数,这里必须是奇数
        self.d,remainder=self.exgcd(self.e,phi) # 私钥
        while remainder!=1:
            self.e+=2
            self.d,remainder=self.exgcd(self.e,phi)
    
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

    def decryption(self,message):
        message=self.power(message,self.d,self.module)
        # return binascii.unhexlify(bytes(hex(message),encoding='utf8')[2:])
        res=[]
        while message:
            res.append(chr(message&255))
            message>>=8
        return ''.join(res[::-1])

class DSA(Base):
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
        s=(Hm+self.x*r)*self.exgcd(k,self.q)[0]
        return r,s

    def check(self,message,r,s):
        w=self.exgcd(s,self.q)[0]
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