# 给定币值n,找出由coins=[1,2,5,10]硬币的所有组合数
def coin_combination(n):
    coins=[1,2,5,10]
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
