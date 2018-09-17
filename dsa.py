import random,hashlib

class DSA:
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

    def sign(self,message):
        k=random.randrange(1,self.q)
        r=pow(self.g,k,self.p)%self.q
        Hm=int(hashlib.sha256(message.encode('utf8')).hexdigest(),16)
        s=(Hm+self.x*r)*self.exgcd(k,self.q)[0]
        return r,s

    def check(self,message,r,s):
        w=self.exgcd(s,self.q)[0]
        Hm=int(hashlib.sha256(message.encode('utf8')).hexdigest(),16)
        u1=Hm*w%self.q
        u2=r*w%self.q
        v=pow(self.g,u1,self.p)*pow(self.y,u2,self.p)%self.p%self.q
        return v==r

    @staticmethod
    def is_probable_prime(n, trials = 10): # Miller-Rabin检测
        assert n > 1
        if n == 2:
            return True
        if not n&1:
            return False
        s = 0
        d = n - 1
        while not d&1:
            s+=1
            d>>=1
         
        for i in range(trials):
            a = random.randrange(2, n)
            if pow(a, d, n) != 1:
                for r in range(s):
                    if pow(a, 2 ** r * d, n) == n - 1:
                        break
                else:
                    return False
        return True

    @staticmethod
    def exgcd(a,b):
        def _exgcd( a , b ):
            if b:
                x , y , remainder = _exgcd( b , a % b )
                x , y = y, ( x - (a // b) * y )
                return x, y, remainder
            return 1, 0, a
        x,y,remainder=_exgcd(a,b)
        while x<0:
            x+=b
        return x,remainder

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

if __name__ == "__main__":
    dsa=DSA()
    message='avatar'
    r,s=dsa.sign(message)
    print(dsa.check(message,r,s))  # True
