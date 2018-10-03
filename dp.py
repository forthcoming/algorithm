# 给定币值n,找出由coins=[1,2,5,10]硬币的所有组合数
def coin(n):
    coins=[1,2,5,10]
    dp=[0]*(n+1)
    dp[0]=1
    for coin in coins:  # 第i次循环后,dp[j]的值为用前i种硬币组成金额j的方法数,复杂度是O(mn),m是多少种钱币
        for idx in range(coin,n+1):
            dp[idx]+=dp[idx-coin] # dp[j]由{不含v[i]币值,含至少一个v[i]}组成
    print(dp[n])
