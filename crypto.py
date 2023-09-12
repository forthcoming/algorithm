import binascii
import hashlib
import random

import numpy as np

'''
模运算：
(a + b) % p = (a % p + b % p) % p
(a - b) % p = (a % p - b % p) % p
(a * b) % p = (a % p * b % p) % p   
a ^ b % p = ((a % p)^b) % p

辗转相除法: 
对整数a,b进行辗转相除,可得它们的最大公约数gcd(a,b),然后收集辗转相除法中产生的式子,倒回去可以得到ax+by=gcd(a,b),其中x,y也是整数
如果gcd(a, b) = 1,則稱a和b互质,a和b是否互质和它们是否质数无关,最小公倍数(a,b)=a*b/gcd(a,b)

欧拉函数:
比n小但与n互质的正整数个数φ(n)称为n的欧拉函数
对任一质数p,有φ(n)＝p-1,对于两个不同的质数p和q则φ(pq)=φ(p)*φ(q)=(p-1)(q-1)

欧拉定理: 如果两个正整数a和b互质,b的欧拉函数φ(b),则a^φ(b) % b = 1

模反元素(前提是互质):
欧拉定理可以用来证明模反元素必然存在,a^(φ(b)-1)就是a对模数b的模反元素,比如3和11互质,则3**(φ(11)-1)=3**9=19683是3对11的模反元素
如果正整数a和b互质,那么一定可以找到整数x,使得ax-1被b整除(可用辗转相除法矩阵乘法证)
ax+by=1,则a,b互质
(ax+by)%b=1%b 
ax%b=1 
即求出a对b的模反元素x
如果x是a对b的模反元素,则x+kb都是a对b的模反元素,比如3和11互质,那么3的模反元素4,4+11=15都满足


RSA步骤:
1. 随机选择两个不相等的大质数p和q,比如61和53; 不相等大质数除了保证因式分解困难外,还确保了对任意0<=m<n都满足m^(ed)%n = m
2. 计算n = p*q = 61×53 = 3233,n的长度就是密钥长度,3233写成二进制是110010100001,一共有12位,所以这个密钥就是12位,实际应用中,RSA密钥一般是1024位或2048位
3. 计算φ(n) = (p-1)(q-1) = 60*52 = 3120
4. 随机选择一个整数e,条件是1< e < φ(n)且e与φ(n)互质,比如e = 17
5. 计算e对于φ(n)的模反元素d,即满足ed%φ(n) = 1,这个方程可以用"扩展欧几里得算法"求解,如d=2753
6. 将n和e封装成公钥,n和d封装成私钥,一旦d泄漏就等于私钥泄漏
7. 加密名文m=65需要用到公钥(n,e),密文c=m^e%n=65^17%3233=2790;注意m必须是整数(字符串可以取ascii值或unicode值)且必须小于n(由加密解密证明可以看到m必须小于n)
8. 解密密文c=2790需要用到私钥(n,d),明文m=c^d%n=2790^2753%3233=65

说明:
公钥和私钥完全对等,所以也可以用私钥加密,公钥解密来做签名
n,e已知,要想知道明文m需要知道d,要想知道d必须知道φ(n),就必须对n做因式分解,但大整数分解质因数很困难,因此安全性得到保证
DSA和RSA都是非对成加密,不同之处在于DSA不能用作加密和解密,也不能进行密钥交换,只用于签名,它比RSA要快很多

RSA加密解密证明:
即证明m = c^d%n,又因为m^e%n = c,转化为证m = m^(ed)%n
又因为ed = hφ(n)+1,转化为证m = m^(hφ(n)+1)%n   ToBeDone...

Base64编码
它是一种基于用64个可打印字符来表示二进制数据的表示方法
Base64一般用于在HTTP协议下传输二进制数据,由于HTTP协议是文本协议,所以在HTTP协议下传输二进制数据需要将二进制数据转换为字符数据,然而直接转换是不行的,因为网络传输只能传输可打印字符
当需要转换的字符不是3的倍数时,一律采用补0的方式凑足3的倍数
Base64适用于小段内容的编码,比如数字证书签名、Cookie的内容等,不能用于加密

消息摘要算法
主要特征是加密过程不需要密钥,并且经过加密的数据无法被解密,只有输入相同的明文数据经过相同的消息摘要算法才能得到相同的密文
消息摘要算法不存在密钥的管理与分发问题,适合于分布式网络上使用,无论输入的消息有多长,计算出来的消息摘要的长度总是固定的
著名的摘要算法有RSA公司的MD5和SHA-1(判断某个文件是否被篡改,或者加盐存储用户密码)

数字签名算法
RSA既可以用公钥加密然后私钥解密,也可以用私钥加密然后公钥解密
RSA适用于少量数据加密,应为速度慢大量数据还是需要对称加密算法(密钥可通过RSA传输)
公钥加密然后私钥解密,可以用于通信中拥有公钥的一方向拥有私钥的另一方传递机密信息,不被第三方窃听
私钥加密然后公钥解密是用在数字签名,因为RSA中的每一个公钥都有唯一的私钥与之对应,任一公钥只能解开对应私钥加密的内容
由于直接对原消息进行签名有安全性问题,而且原消息往往比较大,所以一般对消息计算其摘要,然后对摘要进行签名
一个具体的RSA签名过程如下：
小明对外发布公钥,并声明对应的私钥在自己手上
小明对消息M计算摘要,得到摘要D
小明使用私钥对D进行签名,得到签名S
将M和S一起发送出去
验证过程如下：
接收者首先对M使用跟小明一样的摘要算法计算摘要,得到D
使用小明公钥对S进行解签,得到D’
如果D和D’相同,那么证明M确实是小明发出的,并且没有被篡改过
'''


class Base:
    @staticmethod
    def sha(text):
        return int(hashlib.sha256(text.encode('utf8')).hexdigest(), 16)  # 16进制的字符串转换成十进制整数

    @staticmethod
    def power(a: int, b: int, m: int):  # a**b%m or pow(a,b,m)
        assert b >= 0
        result = 1
        while b:
            if b & 1:
                result = result * a % m  # 防止数字过大导致越界
            b >>= 1  # 隐式减去了1
            a = a * a % m
        return result % m  # 针对b=0的情况

    @staticmethod
    def miller_rabin(n, trials=10):  # error_rate=.25**trials
        if n == 2:  # 2是质数
            return True
        if n == 1 or n & 1 == 0:  # 这里不用加括号,与c运算符优先级有区别
            return False
        s = 0
        d = n - 1
        while not d & 1:  # n-1=(2^s)*d
            s += 1
            d >>= 1
        for _ in range(trials):
            a = random.randrange(2, n - 1)  # 范围[2,n-2]
            if Base.power(a, d, n) != 1:  # (a^d)%n
                for _s in range(s):
                    if Base.power(a, 2 ** _s * d, n) == n - 1:  # (a^((2^_s)*d))%n
                        break
                else:
                    return False  # 一定不是质数
        return True  # 大概率是质数

    @staticmethod
    def generate_prime():
        prime = random.randrange((1 << 199) + 1, 1 << 300, 2)
        while not Base.miller_rabin(prime):
            prime += 2
        return prime

    @staticmethod
    def extended_gcd(a, b):  # 只有当a,b互质时算出的x才有实际意义
        """
        a'x'+b'y'=gcd(a', b'); a'= b; b'= a%b= a - a // b * b
        ===>
        bx' + (a - a//b *b)y' = gcd(a' , b')
        ===>
        ay' + b(x'- a//b *y') = gcd(a', b') = gcd(a, b) = ax + by,建立x,y与x',y'之间的映射关系
        """

        def _extended_gcd(_a, _b):
            if _b:
                _x, _y, _common_divisor = _extended_gcd(_b, _a % _b)
                _x, _y = _y, _x - (_a // _b) * _y
            else:
                # 当b=0时gcd(a,b) = ax + 0y = ad, y可以是任意数,一般选0, 所以x=1时最大公约数最小,common_divisor=a
                _x, _y, _common_divisor = 1, 0, _a
            return _x, _y, _common_divisor

        x, y, common_divisor = _extended_gcd(a, b)
        while x < 0:  # RSA算法要求d是正数
            x += b
        return x, common_divisor

    @staticmethod
    def extended_gcd_iter(a, b):
        prev_x, x = 1, 0
        prev_y, y = 0, 1
        _b = b
        while b:
            q = a // b
            x, prev_x = prev_x - q * x, x
            y, prev_y = prev_y - q * y, y
            a, b = b, a % b
        while prev_x < 0:
            prev_x += _b
        return prev_x, a

        # 矩阵迭代版缺点是处理大整数溢出
        # """
        # a=q0*b+r1
        # b=q1*r1+r2
        # r1=q2*r2+r3
        # ......
        # rn-1=qn*rn+0
        # a    q0 1   q1 1          qn 1   rn        rn
        #   =       *      ...... *      *     = M *
        # b    1  0   1  0          1  0   0          0
        # 此处的rn即为最大公约数
        # """
        # _b = b
        # matrix = np.eye(2, dtype=np.int64)
        # while b:
        #     matrix = matrix @ np.array([[a // b, 1], [1, 0]])  # 注意不能用*
        #     a, b = b, a % b
        # D = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]  # 计算行列式(值是1或者-1,取决于循环的次数)
        # d = matrix[1][1] // D  # matrix的逆矩阵matrix' = matrix* / D, matrix[1][1]对应matrix*[0][0]
        # while d < 0:
        #     d += _b
        # return d, a

        # 矩阵迭代版缺点是处理大整数溢出
        # """
        # 由ad + by = g; bd1 + a%by1 = g可以得到
        # d   0  1    d1   0  1    0  1          0  1    1
        #   =       *    =       *       * ... *       *
        # y   1 -k1   y1   1 -k1   1 -k2         1 -dn   0
        # 其中kn = a//b, a,b是每次迭代中的a,b,思考为啥最后一项是[1,0]
        # """
        # _b = b
        # matrix = np.eye(2, dtype=np.int64)  # 初始化单位矩阵
        # while b:
        #     matrix = matrix @ np.array([[0, 1], [1, -(a // b)]])  # 注意-(a//b)要加括号
        #     a, b = b, a % b
        # d = matrix[0][0]
        # while d < 0:
        #     d += _b
        # return d, a


class RSA(Base):
    def __init__(self):
        # P,Q在初始化后应当被销毁,防止外泄
        p = self.generate_prime()
        q = self.generate_prime()
        while p == q:
            q = self.generate_prime()
        phi = (p - 1) * (q - 1)
        self.module = p * q  # 公钥
        self.e = random.randrange(3, phi, 2)  # 公钥,跟phi互质的任意数,这里必须是奇数
        self.d, common_divisor = self.extended_gcd(self.e, phi)  # 私钥
        while common_divisor != 1:  # 保证self.e和phi互质
            self.e += 2
            self.d, common_divisor = self.extended_gcd(self.e, phi)

    def encryption(self, text):
        text = int(binascii.hexlify(text.encode()), 16)
        assert (0 <= text < self.module)
        return self.power(text, self.e, self.module)

    def decryption(self, cipher):
        cipher = self.power(cipher, self.d, self.module)
        # return binascii.unhexlify(hex(cipher).encode()[2:])
        res = []
        while cipher:
            res.append(chr(cipher & 255))
            cipher >>= 8
        return ''.join(res[::-1])


class DSA(Base):
    def __init__(self):
        self.q = random.randrange((1 << 159) + 1, 1 << 160, 2)  # 公钥,160bit位奇数里面挑选
        while not self.miller_rabin(self.q):
            self.q += 2

        factor = 1 << 300
        while True:
            self.p = factor * self.q + 1  # 公钥
            if self.miller_rabin(self.p):
                break
            factor += 1

        self.g = self.power(random.randrange(2, self.p - 1), factor, self.p)  # 公钥
        self.x = random.randrange(1, self.q)  # 私钥
        self.y = self.power(self.g, self.x, self.p)  # 公钥

    def sign(self, text):  # (x,g p,q)
        k = random.randrange(1, self.q)
        r = self.power(self.g, k, self.p) % self.q
        hm = DSA.sha(text)
        s = (hm + self.x * r) * self.extended_gcd_iter(k, self.q)[0]
        return r, s

    def check(self, text, r, s):  # (_r,_s,y,g,p,q)
        w = self.extended_gcd_iter(s, self.q)[0]
        hm = DSA.sha(text)
        u1 = hm * w % self.q
        u2 = r * w % self.q
        v = self.power(self.g, u1, self.p) * self.power(self.y, u2, self.p) % self.p % self.q
        return v == r


if __name__ == "__main__":
    rsa = RSA()
    message = 'akatsuki'
    print(rsa.decryption(rsa.encryption(message)))

    dsa = DSA()
    message = 'avatar'
    message_r, message_s = dsa.sign(message)
    assert dsa.check(message, message_r, message_s)  # 代表身份和信息均正确
