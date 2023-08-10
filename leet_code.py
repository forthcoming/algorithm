import collections
import heapq
from collections import deque
from itertools import permutations


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
    result = []
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
                result.append(wanted_str)

    _bfs(0)
    return result


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
                    dp[i] = dp[aj - left]
                else:
                    dp[i] = min(dp[i], dp[aj - left])
        dp[i] += 1
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


def leet_code_13(messages, total_money):  # 最多获得的短信条数
    dp = [0] * (total_money + 1)  # dp[i]代表i元时最多可发送的短信条数
    for i in range(1, total_money + 1):
        for money, message in enumerate(messages, 1):
            if money <= i:
                dp[i] = max(dp[i], dp[i - money] + message)
            else:  # 提前结束循化,可以不处理
                break
    return dp[-1]


def leet_code_14(boxes, width):  # 箱子之字型摆放
    result = [""] * width
    index = 0
    asc = True  # True表示从上到下，False表示从下到上
    for c in boxes:
        if index == -1:
            index = 0
            asc = True
        if index == width:
            index = width - 1
            asc = False
        result[index] += c
        if asc:
            index += 1
        else:
            index -= 1
    return result

    # arr = []
    # reverse = False
    # idx = 0
    # length = len(boxes)
    # while idx < length:
    #     if reverse:
    #         arr.append(boxes[idx + width - 1:idx - 1:-1])
    #     else:
    #         arr.append(boxes[idx:idx + width])
    #     reverse = not reverse
    #     idx += width
    # left = idx - length
    # if left > 0:
    #     if reverse:
    #         arr[-1] += "*" * left  # 填充空白位置
    #     else:
    #         arr[-1] = "*" * left + arr[-1]
    #
    # for step in range(width):
    #     tmp = ''.join(each[step] for each in arr)
    #     print(tmp.strip('*'))


def leet_code_15(n, k):  # 对称美学
    # 将R对应成比特位1,B对应成比特位0,适用于长度比较小的情况
    bit_value = 0b1
    for i in range(n - 1):
        offset = 1 << i
        mask = (1 << (offset << 1)) - 1
        bit_value = ~bit_value << offset & mask | bit_value  # 由位运算性质,从左至右依次执行
    bit_length = 1 << (n - 1)
    return bit_value >> (bit_length - k - 1) & 1


def leet_code_16(n, k):  # 对称美学
    def _find(_n, _k, xor):
        if _n == 1:
            return "blue" if xor else "red"
        mid = 1 << (_n - 2)
        if _k < mid:
            return _find(_n - 1, _k, not xor)
        else:
            return _find(_n - 1, _k - mid, xor)

    return _find(n, k, False)


def leet_code_17(sites):  # 快递业务站(也可以并查集,初始状态每个站点作为一个跟节点)
    count = 0
    cover = set()
    length = len(sites)
    for i in range(length):
        if i not in cover:
            count += 1
        site = sites[i]
        for j in range(i, length):  # 对称矩阵,减少遍历次数
            if site[j] == 1:
                cover.add(j)
    return count


def leet_code_18(nodes):  # 寻找路径
    length = len(nodes)
    min_leaf = float("inf")
    result = None
    path = []

    def _find_path(parent_pos):
        path.append(nodes[parent_pos])
        left_pos = (parent_pos << 1) + 1
        right_pos = left_pos + 1
        left_exist = left_pos < length and nodes[left_pos] != -1  # 存在左节点
        right_exist = right_pos < length and nodes[right_pos] != -1  # 存在右节点
        if left_exist:
            _find_path(left_pos)
        if right_exist:
            _find_path(right_pos)
        if not left_exist and not right_exist:  # 叶子结点
            nonlocal result, min_leaf
            if nodes[parent_pos] < min_leaf:
                result = path[:]
                min_leaf = nodes[parent_pos]
        path.pop()

    if nodes and nodes[0] != -1:
        _find_path(0)
    return result


def leet_code_19(nodes):  # 寻找路径(先找到最小叶子结点,再回溯)
    length = len(nodes)
    min_pos = -1
    for pos, node in enumerate(nodes):
        left_pos = (pos << 1) + 1
        right_pos = left_pos + 1
        if (left_pos >= length or nodes[left_pos] == -1) and (right_pos >= length or nodes[right_pos] == -1):
            if node != -1 and (min_pos == -1 or node < nodes[min_pos]):
                min_pos = pos
    path = deque()
    while min_pos >= 0:
        path.appendleft(nodes[min_pos])
        min_pos = (min_pos - 1) >> 1
    return path


def leet_code_20(tasks):  # 任务调度
    success = []
    while tasks:
        start_time = tasks[0]['start_time']
        max_priority_idx = -1
        for idx, task in enumerate(tasks):
            if task['start_time'] == start_time:
                task['start_time'] += 1
                if max_priority_idx == -1 or task['priority'] > tasks[max_priority_idx]['priority']:
                    max_priority_idx = idx
            else:
                break  # 提前结束提高效率,可以不判断
        if max_priority_idx != -1:  # 注意判断,有执行过任务
            tasks[max_priority_idx]['cost'] -= 1
            if tasks[max_priority_idx]['cost'] == 0:
                success.append([tasks[max_priority_idx]['thread_id'], tasks[max_priority_idx]['start_time']])
                tasks[max_priority_idx:] = tasks[max_priority_idx + 1:]
    return success


def leet_code_21(numbers):  # 分奖金(也可以暴力搜索)
    result = [0] * len(numbers)
    stack = []  # 单调栈
    for idx, number in enumerate(numbers):
        while stack and numbers[stack[-1]] < number:
            little_idx = stack.pop()
            result[little_idx] = (number - numbers[little_idx]) * (idx - little_idx)
        stack.append(idx)
    for idx in stack:  # 不要忘了
        result[idx] = numbers[idx]
    return result


def leet_code_22(numbers, k):  # 最差产品奖(优先队列)
    length = len(numbers)
    queue = [(numbers[i], i) for i in range(k)]
    heapq.heapify(queue)
    result = [queue[0][0]]
    for i in range(k, length):
        heapq.heappush(queue, (numbers[i], i))
        while queue[0][1] <= i - k:  # 栈顶元素最小,但需要把滑动窗口边界外的元素剔除掉
            heapq.heappop(queue)
        result.append(queue[0][0])
    return result


def leet_code_23(numbers, k):  # 最差产品奖(单调队列)
    length = len(numbers)
    queue = collections.deque()
    for i in range(k):
        while queue and numbers[i] <= numbers[queue[-1]]:
            queue.pop()
        queue.append(i)

    result = [numbers[queue[0]]]
    for i in range(k, length):
        while queue and numbers[i] <= numbers[queue[-1]]:
            queue.pop()
        queue.append(i)
        if queue[0] <= i - k:
            queue.popleft()
        result.append(numbers[queue[0]])
    return result


def leet_code_24(matrix):  # 查找单入口空闲区域
    def _find_zone(x, y):
        nonlocal entrance, count
        count += 1
        matrix[x][y] = "X"  # 遍历过的点标记为"X",重要!!
        if x == 0 or x == m - 1 or y == 0 or y == n - 1:
            entrance.append((x, y))
            if len(entrance) > 1:  # 只能有一个入口
                return
        # 只往下往右遍历,重要!!
        if x + 1 < m and matrix[x + 1][y] == "O":
            _find_zone(x + 1, y)
        if y + 1 < n and matrix[x][y + 1] == "O":
            _find_zone(x, y + 1)

    m = len(matrix)
    n = len(matrix[0])
    entrance_x = entrance_y = max_count = same_count = -1
    for i in range(m):
        for j in range(n):
            count = 0  # 连通区域大小
            entrance = []  # 入口坐标
            if matrix[i][j] == 'O':
                _find_zone(i, j)
                if len(entrance) == 1:
                    if count > max_count:
                        entrance_x, entrance_y = entrance[0]
                        max_count = count
                        same_count = 1
                    elif count == max_count:
                        same_count += 1
    if same_count == 1:
        return entrance_x, entrance_y, max_count
    elif same_count > 1:
        return max_count


def leet_code_25(prices, k, budget):  # 预定酒店
    price_rating = [(price - budget, price) for price in prices]
    price_rating.sort(key=lambda x: (abs(x[0]), x[0]))
    return sorted([item[1] for item in price_rating[:k]])


def leet_code_26(matrix):  # 基站维护最短距离
    min_distance = float("inf")
    for item in permutations(range(1, len(matrix))):
        from_ = 0
        total_distance = 0
        for to in item:
            total_distance += matrix[from_][to]
            from_ = to
        total_distance += matrix[from_][0]
        min_distance = min(min_distance, total_distance)
    return min_distance


def leet_code_27(matrix, volume):  # 最大报酬(0-1背包问题)
    length = len(matrix)
    dp = [[0] * (volume + 1) for _ in range(length + 1)]  # dp[i][j]代表前i件物品放入容量为j的背包能产生的最大价值
    for i in range(1, length + 1):
        for j in range(1, volume + 1):
            v = matrix[i - 1][0]
            if v > j:
                dp[i][j] = dp[i - 1][j]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - v] + matrix[i - 1][1])
    return dp[-1][-1]


def leet_code_28(arr):  # 二进制差异数
    mapping = {}
    for number in arr:
        while True:
            lower_bit = number & -number
            number -= lower_bit
            if not number:
                if lower_bit in mapping:
                    mapping[lower_bit] += 1
                else:
                    mapping[lower_bit] = 1
                break
    length = len(arr)
    total = 0
    for count in mapping.values():
        length -= count
        total += count * length
    return total

    # # 低效版
    # length = len(arr)
    # total = 0
    # for i in range(length):
    #     for j in range(i + 1, length):
    #         if arr[i] ^ arr[j] > arr[i] & arr[j]:
    #             total += 1
    # return total


def leet_code_29(nodes, x, y):  # 查找二叉树节点
    queue = deque([nodes[0], 1])
    count = 0
    flag = False
    while True:  # 认为x,y合法
        node_or_level = queue.popleft()
        if isinstance(node_or_level, int):
            if node_or_level == x:
                flag = True
            queue.append(node_or_level + 1)
        else:
            if flag:
                count += 1
                if count == y + 1:
                    return node_or_level[0]
            for child in node_or_level[1:]:
                queue.append(nodes[child])


def leet_code_30(candidates, target):  # 硬件产品销售方案
    # https://leetcode.cn/problems/combination-sum/description/
    candidates.sort()  # 非必需,用于减少递归次数
    path = []
    length = len(candidates)
    result = []

    def _find(begin, _target):  # 这里的begin参数保证输出结果没有重复,重要!!
        if _target == 0:
            result.append(path)
        for idx in range(begin, length):
            if candidates[idx] <= _target:
                path.append(candidates[idx])
                _find(idx, _target - candidates[idx])
                path.pop()
            else:
                break

    _find(0, target)


if __name__ == "__main__":
    assert leet_code_1([(3, 9, 2), (4, 7, 3)]) == 5
    assert leet_code_2(3, [3, 2, 2, 1]) == 3
    assert leet_code_3(4, [[1, 2], [2, 4], [3, 4], [1, 3]]) == 7
    assert leet_code_4(10, 20) == 7
    assert leet_code_5('78', 'ux') == ['uw', 'vw', 'vx']
    assert leet_code_6(5, 4, 1) == 0
    assert leet_code_7([[1, 3, 4, 5, 8], [2, 3, 6, 7, 1]], 6, 2) == 4
    assert leet_code_8([3, 7, 40, 10, 60]) == 37
    assert leet_code_9([[0, 2], [4, 6], [8, 10], [1, 9], [1, 5], [5, 9]], 0, 10) == 3
    assert leet_code_9([[1, 4], [2, 5], [3, 6]], 1, 6) == 2
    assert leet_code_10(['b', 'ereddred', 'bw', 'bww', 'bwwl', 'bwwlm', 'bwwln']) == 'bwwln'
    assert leet_code_11([7, 1, 5, 3, 6, 4]) == 5
    assert leet_code_12([7, 1, 5, 3, 6, 4]) == 7
    assert leet_code_13([10, 20, 30, 40, 60, 60, 70, 80, 90, 150], 15) == 210
    assert leet_code_14('abcdefg', 3) == ['afg', 'be', 'cd']
    assert leet_code_15(1, 0) == 1
    assert leet_code_16(64, 73709551616) == 'red'
    assert leet_code_17([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
    ]) == 3
    assert leet_code_18([5, 9, 8, -1, -1, 7, -1, -1, -1, -1, -1, 6]) == [5, 8, 7, 6]
    assert leet_code_19([3, 5, 7, -1, -1, 2, 4]) == deque([3, 7, 2])
    assert leet_code_20([
        {'thread_id': 1, 'priority': 3, 'cost': 5, 'start_time': 1},  # 任务id,任务优先级,执行时间,到达时间
        {'thread_id': 2, 'priority': 1, 'cost': 5, 'start_time': 10},
        {'thread_id': 3, 'priority': 2, 'cost': 7, 'start_time': 12},
        {'thread_id': 4, 'priority': 3, 'cost': 2, 'start_time': 20},
        {'thread_id': 5, 'priority': 4, 'cost': 9, 'start_time': 21},
        {'thread_id': 6, 'priority': 4, 'cost': 2, 'start_time': 22},
    ]) == [[1, 6], [3, 19], [5, 30], [6, 32], [4, 33], [2, 35]]
    assert leet_code_21([2, 10, 3]) == [8, 10, 3]
    assert leet_code_22([12, 3, 8, 6, 5], 3) == [3, 3, 5]
    assert leet_code_23([12, 3, 8, 6, 5], 3) == [3, 3, 5]
    assert leet_code_24([
        ["X", "X", "X", "X"],
        ["X", "O", "O", "X"],
        ["X", "O", "O", "X"],
        ["X", "O", "X", "X"],
    ]) == (3, 1, 5)
    assert leet_code_24([
        ["X", "X", "X", "X"],
        ["X", "O", "O", "O"],
        ["X", "X", "X", "X"],
        ["X", "O", "O", "O"],
        ["X", "X", "X", "X"],
    ]) == 3
    assert leet_code_25([10, 9, 8, 7, 6, 5, 4, 3, 2, 1], 4, 6) == [4, 5, 6, 7]
    assert leet_code_26([
        [0, 2, 1],
        [1, 0, 2],
        [2, 1, 0],
    ]) == 3
    assert leet_code_27([
        [1, 2],
        [2, 4],
        [3, 4],
        [4, 5],
    ], 5) == 8
    assert leet_code_28([7, 6, 33, 2, 1, 9, 88, 4, 3, 5, 2]) == 46
    assert leet_code_29([(10, 1, 2), (-21, 3, 4), (23, 5), (14,), (35,), (66,)], 1, 1) == 23
    leet_code_30([3, 7, 6, 2], 7)
