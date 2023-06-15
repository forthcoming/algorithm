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


def max_product_subarray_dac(arr, start, end) -> float:  # 最大连续积子序列(分治法divide and conquer),复杂度O(nlogn)
    if start == end:
        return arr[start]
    mid = (start + end) >> 1
    max_left = max_product_subarray_dac(arr, start, mid)
    max_right = max_product_subarray_dac(arr, mid + 1, end)
    max_middle = cur_max = cur_min = arr[mid] * arr[mid + 1]
    # 考虑乘积=0可以进一步优化
    for idx in range(mid - 1, -1, -1):  # 这里需要注意数组切片的坑,如果mid-1小于0,仍然会切片
        element = arr[idx]
        if element < 0:
            cur_max, cur_min = cur_min, cur_max
        cur_max *= element
        cur_min *= element
        max_middle = max(max_middle, cur_max)
    for each in arr[mid + 2:]:
        if each < 0:
            cur_max, cur_min = cur_min, cur_max
        cur_max *= each
        cur_min *= each
        max_middle = max(max_middle, cur_max)
    return max(max_left, max_right, max_middle)


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
    dp = [1] * length  # dp[i]意思是以arr[i]结尾的最长递增子序列长度
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


def longest_common_subseq(string_x, string_y):  # 最长公共子序列,O(m*n)
    len_x = len(string_x)
    len_y = len(string_y)
    dp = [[0] * (len_y + 1) for _ in range(len_x + 1)]  # d[i][j]指子串string_x[0:i]与string_y[0:j]的最长公共子序列长度
    for i in range(len_x):
        for j in range(len_y):
            if string_x[i] == string_y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                """
                假设a = dp[i,j], b = dp[i-1,j], c = dp[i,j-1]
                (a>=b) && (a>=c) && (x[i] != y[j]) =>
                (a>=b) && (a>=c) && ((a<=b) || (a<=c)) =>
                ((a>=b) && (a>=c) && (a<=b)) || ((a>=b) && (a>=c) && (a<=c)) =>
                ((a==b) && (a>=c)) || ((a==c) && (a>=b)) =>
                a=max(b,c)
                """
                dp[i + 1][j + 1] = max(dp[i + 1][j], dp[i][j + 1])
    result = []
    while len_x and len_y:  # 最长公共子序列不止一个,回溯的不同方向能找到所有解
        if dp[len_x][len_y] == dp[len_x - 1][len_y]:
            len_x -= 1
        elif dp[len_x][len_y] == dp[len_x][len_y - 1]:
            len_y -= 1
        else:  # 注意这里要求string_x和string_y子串结尾都是满足要求的最后一个字符才行
            len_x -= 1
            len_y -= 1
            # result.append(string_x[len_x])
            result.append(string_y[len_y])
    print(result[::-1])


def longest_common_subseq_recur(string_x, string_y):  # 最长公共子序列带备忘录版递归,O(m+n)
    len_x = len(string_x)
    len_y = len(string_y)
    dp = [[0] * (len_y + 1) for _ in range(len_x + 1)]

    def _solve(x, y, _len_x, _len_y):
        if _len_x and _len_y and not dp[_len_x][_len_y]:
            if x[_len_x - 1] == y[_len_y - 1]:  # dp每个位置不一定都遍历到
                dp[_len_x][_len_y] = _solve(x, y, _len_x - 1, _len_y - 1) + 1
            else:
                dp[_len_x][_len_y] = max(_solve(x, y, _len_x - 1, _len_y), _solve(x, y, _len_x, _len_y - 1))
        return dp[_len_x][_len_y]

    _solve(string_x, string_y, len_x, len_y)
    for i in dp:
        print(i)


def longest_common_subseq_zip(string_x, string_y):  # 最长公共子序列空间压缩法, 待看
    len_x = len(string_x)
    len_y = len(string_y)
    if len_x < len_y:
        string_x, string_y = string_y, string_x
        len_x, len_y = len_y, len_x
    dp = [0] * (len_y + 1)  # longest_common_subseq中dp的最后一行
    for i in range(len_x):
        t = [0, 0]
        for j in range(len_y):
            if string_x[i] == string_y[j]:
                t[1] = dp[j] + 1
            else:
                t[1] = max(t[0], dp[j + 1])
            dp[j] = t[0]
            t[0] = t[1]
        dp[-1] = t[0]
    print(dp)


def longest_common_substring(string_x, string_y):  # 寻找2字符串中的最长公共子串(特殊的子序列)
    x_pos = max_length = 0
    len_x, len_y = len(string_x), len(string_y)
    dp = [[0] * (len_y + 1) for _ in range(len_x + 1)]  # d[i+1][j+1]代表以string_x[i]和string_y[j]结尾最长公共子串
    for i in range(len_x):
        for j in range(len_y):
            if string_x[i] == string_y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
                if dp[i + 1][j + 1] > max_length:
                    max_length = dp[i + 1][j + 1]
                    x_pos = i + 1
            else:
                dp[i + 1][j + 1] = 0
    return string_x[x_pos - max_length:x_pos]

    # x_pos = max_length = 0
    # len_x, len_y = len(string_x), len(string_y)
    # dp = [0] * (len_y + 1)  # 压缩版,还可以以min(len_x,len_y)作为dp长度近一步压缩
    # for i in range(len_x):
    #     pre = 0
    #     for j in range(len_y):
    #         if string_x[i] == string_y[j]:
    #             cur = dp[j] + 1
    #             if cur > max_length:
    #                 max_length = cur
    #                 x_pos = i + 1
    #         else:
    #             cur = 0
    #         dp[j] = pre
    #         pre = cur
    # return string_x[x_pos - max_length:x_pos]

    # len_x, len_y = len(string_x), len(string_y)
    # xj = jpos = max_len = 0
    # matrix = [[i == j for j in string_x] for i in string_y]  # 可以考虑用二进制使其降到一维数组
    # i_start = 0
    # for k in range(len_y - 1, -len_x, -1):
    #     if k < 0:
    #         i_start = -k
    #     i, j = k + i_start, i_start
    #     flag = True
    #     while i < len_y and j < len_x:
    #         if flag and matrix[i][j]:
    #             xj = j
    #             flag = False
    #         elif (not flag) and (not matrix[i][j]):  # 这里注意了，只有flag被改为False才执行
    #             if j - xj > max_len:
    #                 max_len = j - xj
    #                 jpos = xj
    #             flag = True
    #         i += 1
    #         j += 1
    #     if matrix[i - 1][j - 1] and j - xj > max_len:  # 别忘了
    #         max_len = j - xj
    #         jpos = xj
    # return string_x[jpos:jpos + max_len]


if __name__ == "__main__":
    print(max_product_subarray((1, 2, -2, -1, 5, -4)))
    print(max_product_subarray_dac((1, 2, -2, -1, 5, -4), 0, 5))
    print(max_add_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))
    print(coin_number(9, [2, 1, 5]))
    print(coin_change(9, [2, 1, 5]))
    longest_incr_seq([1, -1, 2, -3, 4, -5, 6, -7])
    """
    初始状态(8 X 7)
     [0, 0, 0, 0, 0, 0, 0]
     [0, 0, 0, 0, 0, 0, 0]
     [0, 0, 0, 0, 0, 0, 0]
     [0, 0, 0, 0, 0, 0, 0]
     [0, 0, 0, 0, 0, 0, 0]
     [0, 0, 0, 0, 0, 0, 0]
     [0, 0, 0, 0, 0, 0, 0]
     [0, 0, 0, 0, 0, 0, 0]
    最终状态(8 X 7)
      b  d  c  a  b  a
    a[0, 0, 0, 0, 0, 0, 0]
    b[0, 0, 0, 0, 1, 1, 1]
    c[0, 1, 1, 1, 1, 2, 2]
    b[0, 1, 1, 2, 2, 2, 2]
    d[0, 1, 1, 2, 2, 3, 3]
    a[0, 1, 2, 2, 2, 3, 3]
    b[0, 1, 2, 2, 3, 3, 4]
     [0, 1, 2, 2, 3, 4, 4]
    """
    longest_common_subseq('abcbdab', 'bdcaba')  # ['b', 'c', 'b', 'a']
    longest_common_subseq_zip('bdcaba', 'abcbdab')
    print(longest_common_substring("abcdefghijklmnop", "abcsafjklmnopqrstuvw"))  # jklmnop
