def leet_code_1(tasks: list[tuple[int, int, int]]):  # 最大化控制资源成本
    # tuple[int, int, int]代表[开始时间, 结束时间, 占用资源数]
    max_resource = 0
    borders = [border for task in tasks for border in task[:2]]
    for border in borders:
        _max_resource = 0
        for task in tasks:
            if task[0] <= border <= task[1]:
                _max_resource += task[2]
        max_resource = max(max_resource, _max_resource)
    return max_resource


def leet_code_2(max_weight, weights):  # 骑车去绿岛
    # max_weight: 双人自行车最大承载量
    # weights: 每个人的体重
    # 隐含条件: 一辆双人车最多承载2人,且最大承载量大于每个人的体重
    count = 0
    weights.sort()
    left = 0
    right = len(weights) - 1
    while left < right:
        if weights[left] + weights[right] <= max_weight:
            left += 1
        right -= 1
        count += 1
    if left == right:
        count += 1
    return count


def leet_code_3(point_num, edges):  # 无向图染色
    count = 0
    for i in range(1 << point_num):
        for edge in edges:
            if (i >> (edge[0] - 1) & 1) and (i >> (edge[1] - 1) & 1):
                break
        else:
            count += 1
    return count


def leet_code_4(left, right):  # 不含101的数
    count = 0
    for ele in range(left, right + 1):
        while ele >= 0b101:
            if ele & 0b111 == 0b101:
                break
            ele >>= 1
        else:
            count += 1
    return count


def leet_code_5(num_str, block_str):  # 过滤组合字符串
    mapping = {
        '0': 'abc',
        '1': 'def',
        '2': 'ghi',
        '3': 'jkl',
        '4': 'mno',
        '5': 'pqr',
        '6': 'st',
        '7': 'uv',
        '8': 'wx',
        '9': 'yz',
    }
    stack = []

    def _bfs(idx):
        if idx < len(num_str):
            for char in mapping[num_str[idx]]:
                stack.append(char)
                _bfs(idx + 1)
                stack.pop()
        else:
            wanted_str = ''.join(stack)
            if wanted_str != block_str:
                print(wanted_str)

    _bfs(0)


def leet_code_6(sheep_num, wolf_num, capacity):  # 羊,狼,农夫过河
    min_count = float("inf")

    def _dfs(_sheep_num, _wolf_num, oppo_sheep, oppo_wolf, count):
        if _sheep_num == _wolf_num == 0:
            nonlocal min_count
            min_count = min(min_count, count)
            return
        for s in range(min(_sheep_num, capacity) + 1):
            for w in range(min(_wolf_num, capacity) + 1):
                if s + w == 0:
                    continue
                if 0 < _sheep_num - s <= _wolf_num - w:  # 运输后有羊剩余且羊数量小于等于狼数量
                    continue
                if s + w > capacity:  # 运输的羊+狼大于船容量
                    break
                if 0 < oppo_sheep + s <= oppo_wolf + w:  # 运输后对岸有羊且羊数量小于等于狼数量
                    break
                _dfs(_sheep_num - s, _wolf_num - w, oppo_sheep + s, oppo_wolf + w, count + 1)

    _dfs(sheep_num, wolf_num, 0, 0, 0)
    if min_count == float("inf"):
        min_count = 0
    return min_count


def leet_code_7(matrix, threshold, width):  # 探索地块建立
    x, y = len(matrix), len(matrix[0])
    p = [[0] * (y + 1) for _ in range(x + 1)]  # p[i][j]代表matrix[0][0]与matrix[i-1][j-1]区域的元素和,二维前缀和
    for i in range(x):
        for j in range(y):
            p[i + 1][j + 1] = p[i][j + 1] + p[i + 1][j] - p[i][j] + matrix[i][j]  # 容斥原理

    count = 0
    for i in range(width, x + 1):
        for j in range(width, y + 1):
            if p[i][j] - p[i - width][j] - p[i][j - width] + p[i - width][j - width] >= threshold:
                count += 1
    return count


def leet_code_8(logs):  # 日志首次上报最多积分
    max_score = 0
    minus_score = 0
    left_score = 0
    for current_score in logs:
        if minus_score + current_score <= 100:
            left_score += current_score - minus_score
            max_score = max(max_score, left_score)
            minus_score += current_score
        else:
            break
    return max_score


def leet_code_9(clips: list[list[int]], left, right):  # 区间交叠问题
    # https://leetcode.cn/problems/video-stitching/
    dp = [0] + [-1] * (right - left)
    for i in range(1, 1 + right - left):  # dp[i]代表将区间[left,left+i]覆盖所需的最少子区间数量
        for aj, bj in clips:
            if aj < left + i <= bj:
                if dp[i] == -1:
                    dp[i] = dp[aj - left] + 1
                else:
                    dp[i] = min(dp[i], dp[aj - left] + 1)
    return dp[-1]


def leet_code_10(passwords):  # 最长的密码
    passwords.sort(key=lambda x: (len(x), x))  # 先根据长度排序,再根据字典排序
    passwords_set = set(passwords)
    while passwords:
        password = passwords.pop()
        end = len(password) - 1
        while password[:end] in passwords_set:
            if end == 1:
                return password
            else:
                end -= 1


def leet_code_11(prices: list[int]) -> int:  # 买卖股票的最佳时机I
    # https://leetcode.cn/problems/best-time-to-buy-and-sell-stock
    min_price = prices[0]
    max_profit = 0
    for price in prices[1:]:
        max_profit = max(price - min_price, max_profit)
        min_price = min(price, min_price)
    return max_profit


def leet_code_12(prices: list[int]) -> int:  # 买卖股票的最佳时机II
    # https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/
    length = len(prices)
    # 定义状态dp[i][0]表示第i天交易完后手里没有股票的最大利润,dp[i][1]表示第i天交易完后手里持有一支股票的最大利润(i从0开始)
    dp = [[0, 0] for _ in range(length)]
    dp[0] = [0, -prices[0]]
    for i in range(1, length):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + prices[i])
        dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - prices[i])
    return dp[-1][0]

    # length = len(prices)  # 贪心算法
    # max_profit = 0
    # for i in range(1, length)):
    #     tmp = prices[i] - prices[i - 1]
    #     if tmp > 0:
    #         max_profit += tmp
    # return max_profit


if __name__ == "__main__":
    print(leet_code_1([(3, 9, 2), (4, 7, 3)]))
    print(leet_code_2(3, [3, 2, 2, 1]))
    print(leet_code_3(4, [[1, 2], [2, 4], [3, 4], [1, 3]]))
    print(leet_code_4(10, 20))
    leet_code_5('78', 'ux')
    print(leet_code_6(5, 4, 1))
    print(leet_code_7([[1, 3, 4, 5, 8], [2, 3, 6, 7, 1]], 6, 2))
    print(leet_code_8([3, 7, 40, 10, 60]))
    print(leet_code_9([[0, 2], [4, 6], [8, 10], [1, 9], [1, 5], [5, 9]], 0, 10))
    print(leet_code_9([[1, 4], [2, 5], [3, 6]], 1, 6))
    print(leet_code_10(['b', 'ereddred', 'bw', 'bww', 'bwwl', 'bwwlm', 'bwwln']))
    print(leet_code_11([7, 1, 5, 3, 6, 4]))
    print(leet_code_12([7, 1, 5, 3, 6, 4]))