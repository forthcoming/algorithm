import collections
import heapq
import math
import re
from collections import deque, Counter
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


def leet_code_7(matrix, threshold, width):  # 探索地块建立|荒地建设电站|区域发电量统计
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
    max_score = minus_score = left_score = 0
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
    def _find_zone(x, y):  # 也可以用队列做
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

    m, n = len(matrix), len(matrix[0])
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


def leet_code_27(matrix, volume):  # 最大报酬|查找充电设备组合(0-1背包问题)
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
            result.append(path[:])  # 注意这里要拷贝,不要直接引用
        for idx in range(begin, length):
            if candidates[idx] <= _target:
                path.append(candidates[idx])
                _find(idx, _target - candidates[idx])
                path.pop()
            else:
                break

    _find(0, target)
    return result


def leet_code_31(matrix, x, y):  # 计算网络信号
    m, n = len(matrix), len(matrix[0])
    queue = deque()
    for i in range(m):  # 寻找信号源
        for j in range(n):
            if matrix[i][j] > 0:
                queue.append((i, j))
                break
        else:
            continue
        break

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        i, j = queue.popleft()
        if matrix[i][j] <= 1:
            return 0
        for direction_x, direction_y in directions:
            new_i, new_j = i + direction_x, j + direction_y
            if 0 <= new_i < m and n > new_j >= 0 == matrix[new_i][new_j]:
                matrix[new_i][new_j] = matrix[i][j] - 1
                if new_i == x and new_j == y:
                    return matrix[new_i][new_j]
                queue.append((new_i, new_j))


def leet_code_32(operators, volunteers):  # 核酸最快检测效率(也可以大顶堆做,效率会差一丢丢)
    def _get_incr(op_pos):
        incr = operators[op_pos] * rate
        if with_volunteers[op_pos] == 0:
            incr *= 2
        return incr

    rate = .1
    length = len(operators)
    operators.sort(reverse=True)  # 优先给效率高的采样员配备志愿者
    with_volunteers = [0] * length
    volunteers = min(volunteers, 4 * length)  # 初始化数据(志愿者过多会饱和)

    op_start, op_next = 0, 1
    for v in range(volunteers):
        if op_next >= length:
            with_volunteers[op_start] += 1
            if with_volunteers[op_start] == 4:
                op_start += 1
        else:
            v_start_incr = _get_incr(op_start)
            v_next_incr = _get_incr(op_next)
            if v_start_incr > v_next_incr:
                with_volunteers[op_start] += 1
                if with_volunteers[op_start] == 4:
                    op_start += 1
                    if op_start == op_next:
                        op_next += 1
            else:
                with_volunteers[op_next] += 1
                op_next += 1

    total = 0
    for idx, with_volunteer in enumerate(with_volunteers):
        with_volunteer -= 1 if with_volunteer else 2
        total += (with_volunteer * rate + 1) * operators[idx]
    return int(total)


def leet_code_33(str1, str2):  # 字符串解密
    counter = {}
    length = len(set(str2))
    for word in re.findall(r'[^0-9a-f]+', str1):
        length_word = len(set(word))
        if length_word <= length:
            if length_word in counter:
                counter[length_word].append(word)
            else:
                counter[length_word] = [word]
    if counter:
        max_length = sorted(counter)[-1]
        return sorted(counter[max_length])[-1]


def leet_code_34(string):  # 删除重复数字后的最大数字
    counter = Counter(string)
    used_counter = {char: 0 for char in counter}  # 初始化每个使用过的字符个数为0
    stack = []
    for char in string:
        while stack and counter[stack[-1]] > 2 and stack[-1] < char:
            used_counter[stack[-1]] -= 1
            stack.pop()
            counter[stack[-1]] -= 1
        if used_counter[char] < 2:
            used_counter[char] += 1
            stack.append(char)
    return ''.join(stack)


def leet_code_35(heights):  # 水库蓄水问题(双指针法)
    # https://leetcode.cn/problems/container-with-most-water/description
    length = len(heights)
    left, right = 0, length - 1
    bond_left = bond_right = max_area = -1
    while left < right:
        height = min(heights[right], heights[left])
        area = height * (right - left)
        if area >= max_area:  # 注意这里是大于等于,应为蓄水量一样时要返回距离最近的边界
            bond_left = left
            bond_right = right
            max_area = area
        if heights[right] > heights[left]:
            left += 1
        else:
            right -= 1
    return bond_left, bond_right, max_area


def leet_code_36(arr, k):  # 优雅子数组
    result = 0
    length = len(arr)
    for i in range(length):  # i是子数组起点
        count = {}
        flag = False
        for j in range(i, length):  # j是子数组终点
            key = arr[j]
            if key in count:
                count[key] += 1
            else:
                count[key] = 1
            if count[key] == k:
                flag = True
            if flag:
                if count[key] <= k:
                    result += 1
                else:
                    break
    return result


def leet_code_37(start, end, a, b):  # 数字加减游戏
    """
    (start-end-a*0)%b (start-end+a*0)%b
    (start-end-a*1)%b (start-end+a*1)%b
    ......
    """
    count = 0
    add = sub = end
    while True:
        if (start - add) % b == 0:
            break
        if (start - sub) % b == 0:
            break
        count += 1
        add -= a
        sub += a
    return count


def leet_code_38(fields, days):  # 农场施肥
    left, right = 1, sorted(fields)[-1]
    while left < right:
        mid = (left + right) >> 1
        total = sum(math.ceil(field / mid) for field in fields)
        if total > days:
            left = mid + 1  # 左边一定不满足条件
        else:
            right = mid  # 由于要求最低效率,因此这里不能减1
    if sum(math.ceil(field / left) for field in fields) <= days:  # 此时left==right
        return left
    return -1


def leet_code_39(arr, total):  # 组装新的数组(硬件产品销售方案题目变种)
    count = 0
    arr.sort()  # 元素为正数
    length = len(arr)

    def _find(idx, _total):
        if 0 <= _total < arr[0]:
            nonlocal count
            count += 1
            return
        for i in range(idx, length):
            if arr[i] <= _total:
                _find(i, _total - arr[i])
            else:
                break

    _find(0, total)
    return count


def leet_code_40(task_num, relations):  # 快速开租建站(拓扑排序)
    upstream = [0] * task_num  # 每个任务的前置依赖任务个数
    downstream = [[] for _ in range(task_num)]  # 每个任务的下游任务
    for relation in relations:
        upstream[relation[1]] += 1
        downstream[relation[0]].append(relation[1])

    queue = deque()
    total_duration = 0
    duration = 1
    for task in range(task_num):
        if upstream[task] == 0:
            queue.append((task, duration))
    while queue:
        upstream_task, total_duration = queue.popleft()
        for downstream_task in downstream[upstream_task]:
            upstream[downstream_task] -= 1
            if upstream[downstream_task] == 0:
                queue.append((downstream_task, total_duration + duration))
    return total_duration


def leet_code_41(matrix, task):  # 微服务的集成测试(拓扑排序)
    # length = len(matrix)
    # unfinished_tasks = {*range(length)}
    # total_duration = 0
    # relations = []
    # for i in range(length):
    #     for j in range(length):
    #         if i != j and matrix[i][j]:
    #             relations.append((i, j))
    #
    # while task in unfinished_tasks:
    #     reliance_task = {first_task for first_task, second_task in relations if second_task in unfinished_tasks}
    #     process_task = unfinished_tasks - reliance_task
    #     if task in process_task:
    #         total_duration += matrix[task][task]
    #     else:
    #         total_duration += max(matrix[task][task] for task in process_task)
    #     unfinished_tasks = reliance_task
    # return total_duration

    def _dfs(_task):
        max_time = 0
        for i in range(length):
            if matrix[_task][i] != 0 and i != _task:  # 得到服务k启动依赖的服务
                max_time = max(max_time, _dfs(i))  # 计算启动依赖服务的最大耗时,并记录到总耗时中
        return max_time + matrix[_task][_task]

    length = len(matrix)
    total_duration = _dfs(task)
    return total_duration


def leet_code_42(string):  # 严格递增字符串
    changes = min_changes = string.count("A")
    for char in string:  # 假设字符串长度为3,修改后的字符串一定是BBB,ABB,AAB,AAA中的一种,changes代表开头有0到3个A时需要交换的次数
        # 第几次循环意思是想让前几个字符都变为A
        if char == "A":
            changes -= 1
            min_changes = min(changes, min_changes)
        else:
            changes += 1
    return min_changes


def leet_code_43(img):  # 简单的自动曝光(二分法)
    left, right = -255, 255
    wanted = len(img) << 7
    while left < right:
        mid = (left + right) >> 1
        total = sum(max(0, min(255, mid + pixel)) for pixel in img)  # 单调递增
        if total > wanted:
            right = mid - 1
        elif total < wanted:
            left = mid + 1
        else:
            return mid
    # 此时left==right,left左边均值一定小于128,left右边均值一定大于128,且离left越远相差越大,left对应均值与128比可大可小
    min_delta = float('inf')
    k = -1
    for i in range(left - 1, left + 2):
        delta = abs(sum(max(0, min(255, i + pixel)) for pixel in img) - wanted)
        if delta < min_delta:  # 不能取等号,应为有多个条件满足时取最小值
            k = i
            min_delta = delta
    return k


def leet_code_44(files, capacity):  # 最大连续文件之和|区块链文件转储系统
    # https://leetcode.cn/problems/minimum-size-subarray-sum/description
    # 滑动窗口,前提是连续子数组且元素非负
    if not files:
        return 0
    length = len(files)
    start, end, window_sum, window_max = 0, 0, files[0], 0  # 滑动窗口起始坐标,滑动窗口大小,最大连续文件之和
    while end < length:
        if window_sum < capacity:
            window_max = max(window_max, window_sum)
            end += 1
            if end < length:
                window_sum += files[end]
        elif window_sum == capacity:
            window_max = capacity
            break
        else:
            window_sum -= files[start]
            start += 1
    return window_max


def leet_code_45(content, word):  # 发现新词的数量|新词挖掘
    result = 0
    need = collections.Counter(word)
    word_len = missing = len(word)
    for end, char in enumerate(content):
        if end >= word_len:
            discard_char = content[end - word_len]
            if need[discard_char] >= 0:
                missing += 1
            need[discard_char] += 1
        if need[char] > 0:
            missing -= 1
        need[char] -= 1
        if not missing:  # 当全部覆盖子串时收缩窗口
            result += 1
    return result


def leet_code_46(content, word):  # 最小覆盖子串(滑动窗口)
    # https://leetcode.cn/problems/minimum-window-substring/description
    content_len = len(content)
    need = collections.Counter(word)
    missing = len(word)
    left, right = 0, content_len  # 最小覆盖子串索引位置
    start = 0  # 滑动窗口起始位置
    for end, char in enumerate(content):
        if need[char] > 0:
            missing -= 1
        need[char] -= 1
        if not missing:  # 当全部覆盖子串时收缩窗口(一旦归零不再增加)
            while need[content[start]] < 0:
                need[content[start]] += 1
                start += 1
            if end - start < right - left:
                left, right = start, end
    return content[left:right + 1] if right < content_len else ''

    # 没有上面简洁,但逻辑更清晰些
    # if not content:
    #     return 0
    # start = end = 0
    # content_len = len(content)
    # word_map = collections.Counter(word)
    # word_char_kind = len(word_map)
    # content_map = {content[0]: 1}
    # min_left, min_right = 0, content_len
    # content_char_kind = 1 if content[0] in word_map and word_map[content[0]] == 1 else 0
    # while end < content_len:
    #     if content_char_kind < word_char_kind:
    #         end += 1
    #         if end < content_len:
    #             content_char = content[end]
    #             if content_char in content_map:
    #                 content_map[content_char] += 1
    #             else:
    #                 content_map[content_char] = 1
    #             if content_char in word_map and word_map[content_char] == content_map[content_char]:
    #                 content_char_kind += 1
    #     else:
    #         while True:
    #             left_char = content[start]
    #             if left_char not in word_map or word_map[left_char] != content_map[left_char]:
    #                 content_map[left_char] -= 1
    #                 start += 1
    #             else:
    #                 if end - start < min_right - min_left:
    #                     min_right, min_left = end, start
    #                 content_map[left_char] -= 1
    #                 start += 1
    #                 content_char_kind -= 1
    #                 break
    #
    # return content[min_left:min_right + 1] if min_right < content_len else ''


def leet_code_47(encrypt: str):  # 简单的解压缩算法
    def _do_repeat(repeat_num):
        substring = ''
        top = stack.pop()
        if top.isalpha():
            stack.append(top * repeat_num)
        elif top == '}':
            while stack[-1] != '{':
                substring = f'{stack.pop()}{substring}'  # 注意顺序
            stack.pop()  # 弹出{
            stack.append(substring * repeat_num)

    length = len(encrypt)
    stack: list[str] = []
    repeat = ''  # 存储数字字符
    idx = 0
    while idx < length:
        if encrypt[idx].isdigit():
            repeat += encrypt[idx]
        else:
            if repeat:
                _do_repeat(int(repeat))
                repeat = ''
            stack.append(encrypt[idx])
        idx += 1
    if repeat:  # 不要忘
        _do_repeat(int(repeat))
    return ''.join(stack)


def leet_code_48(matrix):  # 信号发射与接收(单调队列)
    def _do(x, y, stack):
        while stack and matrix[x][y] > stack[-1]:
            result[x][y] += 1
            stack.pop()
        if stack:
            result[x][y] += 1
            if matrix[x][y] == stack[-1]:
                stack.pop()
        stack.append(matrix[x][y])

    m, n = len(matrix), len(matrix[0])
    result = [[0] * n for _ in range(m)]
    for i in range(m):
        stack_east = []
        for j in range(n):
            _do(i, j, stack_east)
    for j in range(n):
        stack_south = []
        for i in range(m):
            _do(i, j, stack_south)
    return result


def leet_code_49(layouts):  # 机房布局
    count = 0
    length = len(layouts)
    boxes = [False] * length
    for idx, layout in enumerate(layouts):
        if layout == 'M' and not (idx and boxes[idx - 1]):  # 左边没电箱(注意右边此时一定无电箱)
            if idx + 1 < length and layouts[idx + 1] == "I":  # 优先给右边安装电表
                boxes[idx + 1] = True
                count += 1
            elif idx and layouts[idx - 1] == "I":
                boxes[idx - 1] = True
                count += 1
            else:
                count = -1
                break
    return count


def leet_code_50(layouts):  # 机房布局
    count = 0
    length = len(layouts)
    i = 0
    while i < length:
        if layouts[i] == "M":
            if i + 1 < length and layouts[i + 1] == "I":  # 优先将电箱放到机柜右边
                count += 1  # 如果放成功,第i个位置为机柜,i+1的位置为电箱
                i += 2  # 第i+2无论是机柜,还是空位,还是电箱都可以,所以这里i+2
            elif i >= 1 and layouts[i - 1] == "I":  # 右边搞不定搞左边
                count += 1
            else:  # 两边都搞不定,只能返回无法放入电箱
                count = -1
                break
        i += 1
    return count


def leet_code_51(work_orders):  # 工单调度策略
    work_orders.sort(key=lambda x: x[0])  # 先根据工单最迟完成时间排序
    max_score = 0
    priority_queue = []
    heapq.heapify(priority_queue)
    cur_time = 0
    for end_time, score in work_orders:
        if end_time >= cur_time + 1:  # +1代表完成该任务需要1个单位时间
            heapq.heappush(priority_queue, score)
            max_score += score
            cur_time += 1
        else:
            if priority_queue:
                min_score = priority_queue[0]  # 哪个更有价值
                if score > min_score:  # 如果当前值更有价值score大于堆顶,就放弃堆顶,完成当前的
                    heapq.heappop(priority_queue)
                    heapq.heappush(priority_queue, score)
                    max_score += score - min_score
    return max_score


def leet_code_52(work_orders):  # 工单调度策略
    work_orders.sort(key=lambda x: -x[1])  # 先根据积分逆序排序
    scores = {}
    for end_time, score in work_orders:
        while end_time and end_time in scores:
            end_time -= 1
        if end_time:
            scores[end_time] = score  # end_time之前任意时刻完成都能得到分数,为保证总分最大,所以按最晚时刻完成
    return sum(scores.values())


def leet_code_53(matrix, t, c):  # 上班之路(也可以递归做)
    find = False
    home = company = (-1, -1)
    m, n = len(matrix), len(matrix[0])
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == 'S':
                home = (i, j)
            elif matrix[i][j] == 'T':
                company = (i, j)

    directions = [(-1, 0, 1), (1, 0, 2), (0, -1, 3), (0, 1, 4)]
    queue = deque([(home, 0, t, c)])
    while queue:
        position, last_direction, t, c = queue.popleft()
        if position == company:
            find = True
            break
        matrix[position[0]][position[1]] = 'V'  # 标记访问过的节点
        for direction in directions:
            x = position[0] + direction[0]
            y = position[1] + direction[1]
            if 0 <= x < m and 0 <= y < n and matrix[x][y] != 'V':
                new_t, new_c = t, c
                if last_direction and last_direction != direction[2]:
                    new_t -= 1
                if matrix[x][y] == '*':
                    new_c -= 1
                if new_t >= 0 and new_c >= 0:
                    queue.append(((x, y), direction[2], new_t, new_c))
    return find


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
    assert leet_code_30([5, 2, 3], 8) == [[2, 2, 2, 2], [2, 3, 3], [3, 5]]
    assert leet_code_31([
        [0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, -1, 4, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0]
    ], 1, 4) == 2
    assert leet_code_32([300, 200, 400, 10], 10) == 1138
    assert leet_code_33("123admyffc79ptaagghi2222smeersst88mnrt", "ssyyfgh") == "mnrt"
    assert leet_code_33("abcmnq", "rt") is None
    assert leet_code_34("54457950451") == '54795041'
    assert leet_code_35([1, 8, 6, 2, 5, 4, 8, 3, 7]) == (1, 8, 49)
    assert leet_code_36([1, 2, 3, 1, 2, 3, 1], 2) == 9
    assert leet_code_37(11, 33, 4, 10) == 2
    assert leet_code_38([5, 7, 9, 15, 10], 4) == -1
    assert leet_code_38([5, 7, 9, 15, 10], 6) == 10
    assert leet_code_38([5, 7, 9, 15, 10], 50) == 1
    assert leet_code_39([2, 3], 5) == 2
    assert leet_code_40(5, [[0, 4], [1, 2], [1, 3], [2, 3], [2, 4]]) == 3
    assert leet_code_41([
        [1, 0, 0, 0, 0],
        [0, 2, 0, 0, 0],
        [1, 1, 3, 0, 0],
        [1, 1, 0, 4, 0],
        [0, 0, 1, 1, 5]
    ], 2) == 5
    assert leet_code_41([
        [2, 0, 0, 0],
        [0, 3, 0, 0],
        [1, 1, 4, 0],
        [1, 1, 1, 5],
    ], 3) == 12
    assert leet_code_42("BAABBABBAB") == 3
    assert leet_code_43([151, 154, 255, 199, 24, 14, 70, 248, 170, 3]) == -1
    assert leet_code_43([90, 211, 64, 178, 90, 48, 106, 187, 57, 134]) == 11
    assert leet_code_44([100, 300, 500, 400, 400, 150, 100], 1000) == 950
    assert leet_code_44([100, 500, 400, 150, 500, 100], 1000) == 1000
    assert leet_code_45('qweebaewqd', 'qwe') == 2
    assert leet_code_46('ADOBECODEBANC', 'ABC') == 'BANC'
    assert leet_code_47('{a3b1{c}3}3') == 'aaabcccaaabcccaaabccc'
    assert leet_code_48([
        [2, 5, 4, 3, 2, 8],
        [9, 7, 5, 10, 10, 3],
    ]) == [[0, 1, 1, 1, 1, 4], [1, 2, 2, 4, 2, 2]]
    assert leet_code_48([[2, 4, 1, 5, 3, 3]]) == [[0, 1, 1, 2, 1, 1]]
    assert leet_code_49("IMMII") == leet_code_50("IMMII") == 2
    assert leet_code_51([
        [1, 6],  # [t,s]意思是工单必须在t时刻之前完成才有s积分,超时也必须完成但没有积分,每个工单需要1小时
        [1, 7],
        [3, 2],
        [3, 1],
        [2, 4],
        [2, 5],
        [6, 1],
    ]) == 15
    assert leet_code_52([
        [1, 6],
        [1, 7],
        [3, 2],
        [3, 1],
        [2, 4],
        [2, 5],
        [6, 1],
    ]) == 15
    assert leet_code_53([
        ['.', '.', 'S', '.', '.'],
        ['*', '*', '*', '*', '.'],
        ['T', '.', '.', '.', '.'],
        ['*', '*', '*', '*', '.'],
        ['.', '.', '.', '.', '.'],
    ], 2, 0) is True
    assert leet_code_53([
        ['.', '*', 'S', '*', '.'],
        ['*', '*', '*', '*', '*'],
        ['.', '.', '*', '.', '.'],
        ['*', '*', '*', '*', '*'],
        ['T', '.', '.', '.', '.'],
    ], 1, 2) is False
