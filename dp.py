# 求连续子列和的最大值
def max_subarray(li):
    value=0
    maximum=-float('inf')
    for i in li:
        value+=i
        if value>maximum:
            maximum=value
        if value<0:
            value=0
    return maximum
    
def max_subarray(li):
    dp = maximum = li[0]
    for x in li[1:]:
        dp = max(x, dp + x)
        maximum = max(maximum, dp)
    return maximum

# 给定币值n,找出由coins=[1,2,5,10]硬币的所有组合数
def coin_combination(n):
    coins=[1,2,5,10] # 硬币大小可随意排列,但初始化dp[0]=1,其余为0
    dp=[0]*(n+1)
    dp[0]=1
    for coin in coins:  # 第i次循环后,dp[j]的值为用前i种硬币组成金额j的方法数,复杂度是O(mn),m是多少种钱币
        for idx in range(coin,n+1):
            dp[idx]+=dp[idx-coin] # dp[j]由{不含v[i]币值,含至少一个v[i]}组成
    print(dp[n])
'''
跳台阶问题
一个台阶总共有n级,如果一次可以跳1级,也可以跳2级,求总共有多少跳法
我们把n级台阶时的跳法看成是n的函数记为f(n),当n>2时第一次跳的时候就有两种不同的选择
一是第一次只跳1级,此时跳法数目等于后面剩下的n-1级台阶的跳法数目,即为f(n-1)
一种选择是第一次跳2级,此时跳法数目等于后面剩下的n-2级台阶的跳法数目,即为f(n-2)
因此n级台阶时的不同跳法的总数f(n)=f(n-1)+f(n-2),上述问题就是我们平常所熟知的Fibonacci数列问题
如果一个人上台阶可以一次上1个,2个或者3个呢
f(1)=1
f(2)=2
f(3)=4
......
f(n)=f(n-1)+f(n-2)+f(n-3) n > 3
'''

# 给定币值n,找出由coins=[1,3,5]硬币的最小组合数
def coin_number(n):
    coins=[1,3,5]
    dp=[0,1,2,1,2,1]
    for idx in range(6,n+1):
        dp.append(min(dp[idx-1],dp[idx-2],dp[idx-5])+1)
    print(dp[-1])

def LCS(x='abcbdab',y='bdcaba'):  # O(m*n)
    xlen=len(x)
    ylen=len(y)
    dp=[[0]*(ylen+1) for i in range(xlen+1)]
    for i in range(xlen):
        for j in range(ylen):
            if x[i]==y[j]:
                dp[i+1][j+1]=dp[i][j]+1
            else:
                dp[i+1][j+1]=max(dp[i+1][j],dp[i][j+1])
    result=[]
    while xlen and ylen: # 最长公共子序列不止一个,回溯的不同方向能找到所有解
        if dp[xlen][ylen]==dp[xlen-1][ylen]:
            xlen-=1
        elif dp[xlen][ylen]==dp[xlen][ylen-1]:
            ylen-=1
        else:
            xlen-=1
            ylen-=1
            result.append(x[xlen])
    print(result[::-1])
        
# 带备忘录版递归
def LCS(x='abcbdab',y='bdcaba'):  # O(m+n)
    xlen=len(x)
    ylen=len(y)
    dp=[[0]*(ylen+1) for i in range(xlen+1)]
    def _solve(x,y,xlen,ylen):
        if xlen and ylen and not dp[xlen][ylen]:
            if x[xlen-1]==y[ylen-1]:
                dp[xlen][ylen]=_solve(x,y,xlen-1,ylen-1)+1
            else:
                dp[xlen][ylen]=max(_solve(x,y,xlen-1,ylen),_solve(x,y,xlen,ylen-1))
        return dp[xlen][ylen]
    _solve(x,y,xlen,ylen)
    for i in dp:
        print(i)
        
# 空间压缩法(没看懂)
def LCS(x='abcbdab',y='bdcaba'):
    if len(x)<len(y):
        x,y=y,x
    dp=[0]*(len(y)+1)
    for i in range(len(x)):
        t=[0,0]
        for j in range(len(y)):
            if x[i]==y[j]:
                t[1]=dp[j]+1
            else:
                t[1]=max(t[0],dp[j+1])
            dp[j]=t[0]
            t[0]=t[1]
        dp[-1]=t[0]
    print(dp)

def LIS(li=[1,-1,2,-3,4,-5,6,-7]): # O(n^2),还可以将li排序得到新的数组与原数组构成LCS问题
    if not li:
        return -1
    maximum=1
    length=len(li)
    dp=[1]*length
    path=[-1]*length
    for idx in range(1,length):
        for _idx in range(idx):
            if li[idx]>=li[_idx] and dp[idx]<dp[_idx]+1:
                dp[idx]=dp[_idx]+1
                path[idx]=_idx
        if maximum<dp[idx]:
            maximum=dp[idx]
    
    for idx in range(length):
        result=[li[idx]]
        while path[idx]!=-1:
            idx=path[idx]
            result.append(li[idx])
        print(result[::-1])
    return maximum

'''
多源最短路算法时间复杂度是O(V^3),适用于邻接矩阵表示的稠密图,稀疏图则可以迭代调用Dijkstra函数V次即可
graph=[
    [0,2,6,4],
    [float('inf'),0,3,float('inf')],
    [7,float('inf'),0,1],
    [5,float('inf'),12,0],
]
'''
def floyd(graph):
    length=len(graph)
    path=[[-1]*length for j in range(length)]
    for k in range(length):  # k要写外面,里面的i,j是对称的,随便嵌套没所谓
        for i in range(length):
            for j in range(length):
                if graph[i][j]>graph[i][k]+graph[k][j]: # 加=不影响graph结果,但会影响path导致路径出错
                    graph[i][j]=graph[i][k]+graph[k][j]
                    path[i][j]=k
    def __show(i,j):
        if path[i][j]==-1:
            print(f'{i}=>{j}',end=' ')
        else:
            __show(i,path[i][j])
            __show(path[i][j],j)
            # print(f'{path[i][j]}=>{j}',end=' ')  # error
    for i in range(length):
        for j in range(length):
            __show(i,j)
            print(f'shortest path is {graph[i][j]}')
            
'''
编号从0开始,每第k个被杀死,队列,环形链表均可实现
0,   1,    2,    ...,k-1,   k,k+1,...,n-1   # 规模n
n-k  n-k+1 n-k+2,...,killed,0,  1,...,n-k-1 # 杀死第k个得规模n-1
f(n, k) = (f(n-1, k) + k) % n 
'''
def josephus(n,k): 
    dp=0
    for idx in range(2,n+1):
        dp=(dp+k)%idx  # 还能优化
    return dp
