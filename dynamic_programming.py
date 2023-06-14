def max_product_subarray(arr) -> float:  # 最大连续积子序列
    if arr is None:
        return float("inf")
    max_product = cur_max = cur_min = arr[0]
    for each in arr[1:]:
        if each < 0:
            cur_max, cur_min = cur_min, cur_max
        cur_max = max(each, cur_max * each)
        cur_min = min(each, cur_min * each)
        if cur_max > max_product:
            max_product = cur_max

        # cur_max *= each
        # cur_min *= each  # 此时的cur_min和cur_max不一定是最大最小值
        # cur_min, cur_max = min(each, cur_min, cur_max), max(each, cur_min, cur_max)  # 如果不连写,需要给cur_min设置临时值
    return max_product


def max_add_subarray(arr):  # 最大连续和子序列
    dp = maximum = -float("inf")
    for each in arr:
        dp = max(each, dp + each)
        maximum = max(maximum, dp)
    return maximum


def coin_number(amount, coins):  # 找出由coins组合面值为amount的最小组合数,背包问题
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    for sub_amount in range(1, amount + 1):
        for coin in coins:
            if sub_amount >= coin:
                dp[sub_amount] = min(dp[sub_amount], dp[sub_amount - coin])
        dp[sub_amount] += 1
    return dp[amount]


def coin_change(amount, coins):  # 找出由coins组合面值为amount的所有组合(注意不是排列问题),背包问题,非常经典
    dp = [0] * (amount + 1)
    dp[0] = 1
    for coin in coins:  # 第i次循环后,dp[sub_amount]的值为用前i种硬币组成金额j的方法数,复杂度是O(mn),m是多少种钱币
        for sub_amount in range(coin, amount + 1):
            dp[sub_amount] += dp[sub_amount - coin]  # 假设coins=[1,2,5],dp[9]可以看做由1,2,5构成的4元所有组合+由1,2构成的9元所有组合
    return dp[amount]


def longest_incr_seq(arr):  # 最长递增子序列,O(n^2),还可以将arr排序得到新的数组与原数组构成longest_common_seq问题
    if not arr:
        return 0
    length = len(arr)
    dp = [1] * length  # dp[i]意思是数组0到i之中最长递增子序列
    path = [-1] * length
    for idx in range(1, length):
        for _idx in range(idx):
            if arr[idx] >= arr[_idx] and dp[idx] < dp[_idx] + 1:
                dp[idx] = dp[_idx] + 1  # 如果最长递增子序列有多个,这里只记录第一次出现
                path[idx] = _idx  # 经典,记录每一个dp[i]前面的第一个元素下标

    print(dp)
    for idx in range(length):
        result = [arr[idx]]
        while path[idx] != -1:
            idx = path[idx]
            result.append(arr[idx])
        print(result[::-1])


def LCS(x='abcbdab', y='bdcaba'):  # O(m*n)
    xlen = len(x)
    ylen = len(y)
    dp = [[0] * (ylen + 1) for _ in range(xlen + 1)]
    for i in range(xlen):
        for j in range(ylen):
            if x[i] == y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    result = []
    while xlen and ylen:  # 最长公共子序列不止一个,回溯的不同方向能找到所有解
        if dp[xlen][ylen] == dp[xlen - 1][ylen]:
            xlen -= 1
        elif dp[xlen][ylen] == dp[xlen][ylen - 1]:
            ylen -= 1
        else:
            xlen -= 1
            ylen -= 1
            result.append(x[xlen])
    print(result[::-1])


# 带备忘录版递归
def LCS(x='abcbdab', y='bdcaba'):  # O(m+n)
    xlen = len(x)
    ylen = len(y)
    dp = [[0] * (ylen + 1) for i in range(xlen + 1)]

    def _solve(x, y, xlen, ylen):
        if xlen and ylen and not dp[xlen][ylen]:
            if x[xlen - 1] == y[ylen - 1]:
                dp[xlen][ylen] = _solve(x, y, xlen - 1, ylen - 1) + 1
            else:
                dp[xlen][ylen] = max(_solve(x, y, xlen - 1, ylen), _solve(x, y, xlen, ylen - 1))
        return dp[xlen][ylen]

    _solve(x, y, xlen, ylen)
    for i in dp:
        print(i)


# 空间压缩法(没看懂)
def LCS(x='abcbdab', y='bdcaba'):
    if len(x) < len(y):
        x, y = y, x
    dp = [0] * (len(y) + 1)
    for i in range(len(x)):
        t = [0, 0]
        for j in range(len(y)):
            if x[i] == y[j]:
                t[1] = dp[j] + 1
            else:
                t[1] = max(t[0], dp[j + 1])
            dp[j] = t[0]
            t[0] = t[1]
        dp[-1] = t[0]
    print(dp)


if __name__ == "__main__":
    print(max_product_subarray((1, 2, -2, -1, 5, -4)))
    print(max_add_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
    print(coin_number(9, [2, 1, 5]))
    print(coin_change(9, [2, 1, 5]))
    longest_incr_seq([1, -1, 2, -3, 4, -5, 6, -7])
