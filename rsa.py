import random,binascii

class RSA:

    def __init__(self):
        P = self.generate_prime()
        Q = self.generate_prime()
        self.module = P * Q
        while True:
            e = 65537  # 跟(P-1)*(Q-1)互质的任意数,这里必须是奇数
            d,remainder=self.exgcd(e,(P-1) * (Q-1))
            if remainder==1:
                self.e=e # 公钥
                self.d=d # 私钥
                break

    @staticmethod
    def is_probable_prime(n, trials = 50): # Miller-Rabin检测,error_rate=.25**trials
        assert n > 1
        if n == 2: # 2是素数
            return True
        if not n&1: # 排除偶数
            return False
        # 把n-1写成(2^s)*d的形式
        s = 0
        d = n - 1
        while not d&1:
            s+=1
            d>>=1
        
        # trials为测试次数，默认测试5次
        # 每次的底a是不一样的，只要有一次未通过测试，则判定为合数
        for i in range(trials):
            a = random.randrange(2, n)
            if pow(a, d, n) != 1: # 相当于(a^d)%n
                for r in range(s):
                    if pow(a, 2 ** r * d, n) == n - 1: #相当于(a^((2^i)*d))%n
                        break
                else:
                    return False  # 以上条件都满足时,n一定是合数
        return True
    
    @staticmethod
    def generate_prime():
        prime=random.randrange((1<<100)-1,1<<200,2)
        while not RSA.is_probable_prime(prime, trials = 50):
            prime+=2
        return prime

    def encode(self,message): 
        '''
        模运算：
        (a + b) % p = (a % p + b % p) % p
        (a - b) % p = (a % p - b % p) % p
        (a * b) % p = (a % p * b % p) % p   
        a ^ b % p = ((a % p)^b) % p
        '''
        message=int(binascii.hexlify(bytes(message,encoding='utf8')),16)
        assert(message<self.module)
        e=self.e
        res=1
        while e:
            if e&1:
                res=res*message%self.module  # 防止数字过大导致越界
            e>>=1 # 隐式的减去了1
            message=message*message%self.module
        return res

    def decode(self,message):
        message=pow(message,self.d,self.module) # message**key[0]%key[1] 
        res=[]
        while message:
            res.append(chr(message&255))
            message>>=8
        return ''.join(res[::-1])

    @staticmethod
    def exgcd(a,b):
        def _exgcd( a , b ):   # 整数a對模数b之模反元素存在的充分必要條件是a和b互質
            if b:
                x , y , remainder = _exgcd( b , a % b )
                x , y = y, ( x - (a // b) * y )
                return x, y, remainder
            return 1, 0, a
            # return 1, 0, 1
        x,y,remainder=_exgcd(a,b)
        while x<0:  # 此时求的是最小的模反元素,还需要将它转换成正数
            x+=b
        return x,remainder

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

if __name__ == "__main__":
    rsa=RSA() 
    message='akatsuki'
    secret=rsa.encode(message)
    print(secret,rsa.decode(secret))
