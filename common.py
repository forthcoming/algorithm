"""
O(1),O(log(n)),O(n^a)等属于多项式级的复杂度,O(a^n)和O(n!)属于非多项式级的复杂度
P问题指可以在多项式时间内解决的问题集合; NP问题指可以在多项式时间内验证问题的解的问题集合
所有P问题都是NP问题,也就是说能多项式地解决一个问题,必能多项式地验证一个问题的解
通常所谓的NP问题,就是证明或推翻P=NP
有不是NP问题的问题,即提供一个解但不能在多项式的时间里验证它,通常只有NP问题才可能找到多项式算法
不可解问题(Undecidable Problems)指没有任何算法可以解决,无论是在有限时间内还是无限时间内,如停机问题（Halting Problem）
约化(reduction)是将一个问题转化为另一个问题的过程,通过约化我们可以利用已有问题的解决方法来解决新问题
约化具有传递性,如果问题A可约化为问题B,问题B可约化为问题C,则问题A一定可约化为问题C,A约化到B后时间复杂度大于等于A,问题的应用范围也增大
可约化是指可"多项式地"约化(Polynomial-time Reducible),即变换输入的方法是能在多项式的时间里完成
如果不断地约化,最后可以找到一个时间复杂度最高,并且能通吃所有NP问题,称为NPC问题或NP完全问题,是一种特殊的NP问题,目前没有多项式的算法
既然所有NP问题都能约化成NPC问题,那么只要任意一个NPC问题找到了一个多项式的算法,那么所有NP问题都能用这个算法解决,NP也就等于P
逻辑电路问题是第一个NPC问题,其它的NPC问题都是由这个问题约化而来
NP-Hard问题指所有NP问题都能约化到它,但它不一定是NP问题,目前同样没有多项式的算法,由于NP-Hard放宽了限定条件，它将有可能比所有的NPC问题的时间复杂度更高从而更难以解决
NPC问题也是NP-Hard问题如旅行商问题(Traveling Salesman Problem)、背包问题(Knapsack Problem)、图着色问题(Graph Coloring Problem)等
"""
import random
from collections import deque
from itertools import permutations

from heap import Heap


def find_min_step(destination):  # destination每步可以+-2,+-3,最少多少次可以到0
    # 也可以动态规划dp[i]=min(dp[i-2],dp[i-3])+1
    if destination == 1:
        return 2
    count = destination // 3
    destination -= count * 3
    if destination:
        count += 1
    return count


def factorization(n):  # 因式分解
    result = ''
    factor = 2
    while n != 1:
        idx = 0
        while not n % factor:
            idx += 1
            n /= factor
        if idx:
            result += f'{factor}**{idx} '
        factor += 1
    print(result)


def josephus(n, k):  # 约瑟夫环问题
    """
    编号从0开始,每第k个被杀死,队列,环形链表均可实现
    0,   1,     2,    ...,k-1,    k, k+1,...,n-2,   n-1   # 规模n
    n-k, n-k+1, n-k+2,...,killed, 0, 1,...,  n-k-2, n-k-1 # 杀死第k个得规模n-1
    当杀掉一个人后会从k作为起始位置继续报数,设新下标为new,旧下标old,则
    new = (old+n-k) % n 或者 old = (new+k) % n
    f(n, k) = (f(n-1, k) + k) % n , f(n,k)意思是n个人报数k的约瑟夫环游戏最终留下谁
    """
    survive = 0  # 只有一个人的时候,留下人的编号
    for _n in range(2, n + 1):
        survive = (survive + k) % _n
    return survive


def range_5_to_12():  # 由随机范围[2,7]得到随机范围[5,12]
    # f = random.randrange(2, 8)等概率產生[2-7], f随机长度是6,range_5_to_12长度是8,f至少需要执行2次才能覆盖[5,12]
    total = (random.randrange(2, 8) - 2) * 6 + random.randrange(2, 8) - 2  # [0,35],两个f()不能合并,必须调用2次
    if 32 <= total <= 35:
        return range_5_to_12()  # 此处可以用for循环代替
    return total // 4 + 5

    # result = float("inf")
    # while result > 12:
    #     if random.randrange(2, 8) < 5:
    #         result = random.randrange(2, 8) + 3
    #     else:
    #         result = random.randrange(2, 8) + 9
    # return result


def young_tableau(arr, element):  # 杨氏矩阵查找
    # 在一个m行n列二维数组中,每一行都按照从左到右递增的顺序排序,每一列都按照从上到下递增的顺序排序,请完成一个函数,输入这样的一个二维数组和一个整数,判断数组中是否含有该整数
    # 以右上角为例,当右上角大于要查找的数字时排除一行,当右上角大于要查找的数字时排除一列
    row = len(arr) - 1
    column = len(arr[0]) - 1
    r = 0
    c = column
    while c >= 0 and r <= row:
        value = arr[r, c]
        if element < value:
            c -= 1
        elif element > value:
            r += 1
        else:
            return True
    return False


def number_hash(number, digit, a=(5 ** .5 - 1) / 2):
    return int(a * number % 1 * digit)  # 数字哈希方法有除留取余法,平方取中法(按比特位取中),折叠法,字符串哈希用rolling hash


def hanoi(n, left='L', middle='M', right='R'):  # 汉诺塔
    if n == 1:
        print('{} --> {}'.format(left, right))
    else:
        hanoi(n - 1, left, right, middle)
        print('{} --> {}'.format(left, right))
        hanoi(n - 1, middle, left, right)


def hanoi_stack(n):
    stack = [(n, 'L', 'M', 'R')]
    while stack:
        number, left, middle, right = stack.pop()
        if number > 1:
            stack.append((number - 1, middle, left, right))
            stack.append((1, left, middle, right))
            stack.append((number - 1, left, right, middle))
        else:
            print('{} --> {}'.format(left, right))


def zeros(n):  # 统计阶乘数n末尾0的个数,实质就是统计[1,n]中含多少个因子5
    step = 5
    cnt = 0
    while step <= n:
        cnt += n // step
        step *= 5
    return cnt


def count_bit_64(number):  # 计算某个自然数中比特位为1的个数是基数还是偶数
    base = 1
    for _ in range(6):
        number ^= number >> base  # 第一次执行完,number最低位表示之前末2位1的奇偶性,第二次执行完,number最低位表示之前末4位1的奇偶性
        base <<= 1
    return number & 1


def alpha_change(alphabet):  # 大小写互转
    five = 1 << 5
    alphabet = ord(alphabet)
    if alphabet & five:
        alphabet &= ~five  # 指定位取0, 0100000101011010 A~Z
    else:
        alphabet |= five  # 指定位取1, 0110000101111010 a~z
    return chr(alphabet)


# 字符串压缩(一串字母(a~z)组成的字符串,将字符串中连续出席的重复字母进行压缩,'qddddftddjhh' => 'q4dft2dj2h')
def encode_string(string):
    result = ""
    pre = ''
    count = 1
    for char in string:
        if char == pre:
            count += 1
        else:
            if count >= 2:
                result = f'{result}{count}{pre}'
            else:
                result = f'{result}{pre}'
            pre = char
            count = 1
    if count >= 2:
        result = f'{result}{count}{pre}'
    else:
        result = f'{result}{pre}'
    return result


def count_leading_zeros(number, length=64):  # 2分查找思想
    left = 0  # 注意这里从0开始
    right = length
    while left < right:  # 不能有等号
        mid = (left + right) >> 1
        if number >> mid:
            left = mid + 1
        else:
            right = mid
    return length - right  # 此时left == right


def binary_search(arr, num):  # 二分查找,序列必须有序,ASL=(n+1)*log(n+1)/n - 1
    left = 0
    right = len(arr)
    while left <= right:
        mid = (left + right) >> 1
        if arr[mid] > num:
            right = mid - 1
        elif arr[mid] < num:
            left = mid + 1
        else:
            return mid
    return -1


def binary_search_recur(arr, left, right, num):
    if left <= right:
        mid = (left + right) >> 1
        if arr[mid] > num:
            right = mid - 1
        elif arr[mid] < num:
            left = mid + 1
        else:
            return mid
        return binary_search_recur(arr, left, right, num)
    else:
        return -1


def cycle_search(arr, target):  # 移位有序数组查找, eg: arr=[6, 7, 8, 9, 0, 1, 2, 3, 4, 5]
    left = 0
    right = len(arr) - 1
    while left <= right:
        mid = (left + right) >> 1
        if arr[mid] == target:
            return mid
        if arr[left] <= arr[mid]:
            if arr[left] <= target < arr[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if arr[mid] < target <= arr[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1


def bisect_right(arr, x):
    """
    Return the index where to insert item x in list arr, assuming arr is sorted.
    if x already appears in the list, arr.insert(x) will insert just after the rightmost x already there.
    """
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) >> 1
        if x < arr[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return low


def bisect_left(arr, x):
    low = 0
    high = len(arr) - 1
    while low <= high:
        mid = (low + high) >> 1
        if x > arr[mid]:
            low = mid + 1
        else:
            high = mid - 1
    return low


def one_to_another(one, another):  # 对于一对正整数one,another,对one只能进行加1,减1,乘2操作,最少进行几次操作得到another,BFS
    visited = {one}  # 重要
    queue = deque(((one, 0),))  # 注意写法
    while queue:
        number, level = queue.popleft()
        if number == another:
            return level
        level += 1
        number_plus = number + 1
        number_minus = number - 1
        number_twice = number << 1
        if number_plus not in visited:
            visited.add(number_plus)
            queue.append((number_plus, level))
        if number_minus not in visited:
            visited.add(number_minus)
            queue.append((number_minus, level))
        if number_twice not in visited:
            visited.add(number_twice)
            queue.append((number_twice, level))


def tow_sum(arr, number):  # 寻找升序数组和为定值的两个数,eg [1,3,4,5,6,7,8,9,10,11]
    begin, end = 0, len(arr) - 1
    while begin < end:
        total = arr[begin] + arr[end]
        if total == number:
            print(begin, end)
            end -= 1
            begin += 1
        elif total > number:
            end -= 1
        else:
            begin += 1


def get_prime(num):  # 筛选素数
    a = [1] * (num + 1)
    for i in range(2, int(num ** .5) + 1):
        if a[i]:
            j = i
            while j * i <= num:
                a[i * j] = 0
                j += 1
    return [i for i in range(2, num + 1) if a[i]]


def pascal_triangle(level):  # 杨辉三角
    queue = deque([1])
    for k in range(level):
        queue.append(0)
        print(' ' * (level - k - 1), end='')
        for _ in range(k + 1):
            cur = queue.popleft()
            cur += queue[0]
            print(cur, end='  ')
            queue.append(cur)
        print()


def catalan_number(m, n):
    """
    m+n个人排队买票,并且满足m≥n,票价为50元,其中m个人有且仅有一张50元钞票,n个人有且仅有一张100元钞票,初始时候售票窗口没有钱,问有多少种排队的情况数能够让大家都买到票

    12个高矮不同的人排成两排,每排必须是从矮到高排列,且第二排比对应的第一排的人高,问排列方式有多少种？
    我们先把这12个人从低到高排列,然后选择6个人排在第一排,那么剩下的6个肯定是在第二排,用0表示对应的人在第一排,用1表示对应的人在第二排,问题转换为这样的满足条件的01序列有多少个

    如果把0看成入栈操作,1看成出栈操作,就是说给定6个元素,合法的入栈出栈序列有多少个? 有N个节点的二叉树共有多少种情形?

    当第m+n是50元时总共有catalan_number(m-1,n),当第m+n是100元时总共有catalan_number(m,n-1),可得递推公式
    catalan_number(m,n) = catalan_number(m-1,n) + catalan_number(m,n-1); catalan_number(m,0) = 1; catalan_number(m,n) = 0,m<n

    通项公式catalan_number(m,n) = C(n,m+n) - C(m+1,m+n)
    h(n) = C(2n,n)/(n+1) = C(2n,n)-C(2n,n-1) = h(n-1)*(4n-2)/(n+1) = h(0)*h(n-1)+h(1)*h(n-2)+...+h(n-1)*h(0)
    利用翻折思想,把第一个不符合要求的地方后面的排列值互换,总体就能得到一个新的排列,且该排列跟原先不符合的排列一一对应

    错排
    n封信装入不同信封,全装错的个数D(n) = (n-1)[D(n-1) + D(n-2)],思想是寻找子结构,其中D(1)=0, D(2)=1
    """
    # result = [[0] * (n + 1) for _ in range(m + 1)]
    # for i in range(m + 1):
    #     for j in range(n + 1):
    #         if j == 0:
    #             result[i][j] = 1
    #         elif i >= j:
    #             result[i][j] = result[i - 1][j] + result[i][j - 1]
    # return result[m][n]

    result = [0] * (n + 1)
    result[0] = 1
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if i >= j:
                result[j] += result[j - 1]
            else:
                result[j] = 0
    return result[-1]


def push_pop(push, out):  # 判断入栈出栈序列是否合法
    stack = []
    i = j = 0
    if len(push) != len(out):
        return False
    while i < len(push):  # 当push和out长度想同时i一定先于j增加到最末位
        if not stack or out[j] != stack[-1]:
            stack.append(push[i])
            i += 1
        if out[j] == stack[-1]:
            j += 1
            stack.pop()
    while stack:
        if stack.pop() == out[j]:
            j += 1
        else:
            return False
    return True

    # stack = []
    # j = 0
    # for i in out:
    #     if stack and stack[-1] == i:
    #         stack.pop()
    #         continue
    #     while j < len(push):
    #         j += 1
    #         if push[j - 1] == i:
    #             break
    #         else:
    #             stack.append(push[j - 1])
    #     else:
    #         return False
    # return True  # 也可以直接return not stack


def sqrt(t, precision=20):
    """
    # Newton法求f(x)=x**4+x-10=0在[1,2]内的一个实根
    x=1  # x也可以是2,不能乱选,对于f(x)=1/(x*x)-t,递推关系式为x *= .5 * (3 - t * x ** 2),t是常数,初始值选取要注意
    for i in range(10):
        x=(3*x**4+10)/(4*x**3+1)
    """
    assert t > 0 and type(precision) == int and precision > 0
    border = t  # border也可以是2t等
    for i in range(precision):
        border = .5 * (border + t / border)  # 牛顿法,收敛速度快,优于二分法
    return border


def python(n):  # 蛇形填数
    """
    假如n=4,则会按步长4,3,3,2,2,1,1沿右下左上方向遍历
    """
    arr = [[0] * n for _ in range(n)]
    arr[0] = list(range(1, n + 1))
    x = 0
    y = n - 1
    step = count = n
    while step > 0:
        step -= 1
        for _ in range(step):
            x += 1
            count += 1
            arr[x][y] = count
        for _ in range(step):
            y -= 1
            count += 1
            arr[x][y] = count
        step -= 1
        for _ in range(step):
            x -= 1
            count += 1
            arr[x][y] = count
        for _ in range(step):
            y += 1
            count += 1
            arr[x][y] = count
    for i in arr:
        print(i)

    # count = 1
    # x = y = 0
    # arr = [[0] * n for _ in range(n)]
    # while count <= n ** 2:
    #     while y < n and not arr[x][y]:
    #         arr[x][y] = count
    #         y += 1
    #         count += 1
    #     y -= 1
    #     x += 1
    #     while x < n and not arr[x][y]:
    #         arr[x][y] = count
    #         x += 1
    #         count += 1
    #     x -= 1
    #     y -= 1
    #     while y >= 0 and not arr[x][y]:
    #         arr[x][y] = count
    #         y -= 1
    #         count += 1
    #     y += 1
    #     x -= 1
    #     while x >= 0 and not arr[x][y]:
    #         arr[x][y] = count
    #         x -= 1
    #         count += 1
    #     x += 1
    #     y += 1
    # for i in arr:
    #     print(i)


def _cmp(little=True):
    if little:  # 小顶堆
        return lambda x, y: x >= y
    else:
        return lambda x, y: x < y


def top_k(arr, k):  # 选择数组中第k个数,构建一个大顶堆; 构建一个大小为k的小顶堆; 快排变形
    length = len(arr)
    assert 1 <= k <= length
    if k <= (length >> 1):
        cmp = _cmp(little=False)
    else:  # 尽可能使堆较小
        k = length - k + 1
        cmp = _cmp(little=True)
    heap = Heap(arr[:k], key=cmp)
    for idx in range(k, length):
        if cmp(arr[idx], heap.top()):
            heap.pop()
            heap.push(arr[idx])
    return heap.top()

    # length = len(arr)
    # assert 1 <= k <= length
    # k = length - k + 1  # 问题转换为小顶堆
    # heap_arr = arr[:k]
    # heapq.heapify(heap_arr)
    # for idx in range(k, length):
    #     if arr[idx] > heap_arr[0]:
    #         heappop(heap_arr)
    #         heappush(heap_arr, arr[idx])
    # return heap_arr[0]


def top_k_quicksort(arr, k):  # 选择数组中第k个数
    length = len(arr)
    index = k - 1

    def _top_k(left, right):
        if left > right:
            return
        mid = left
        for idx in range(left + 1, right + 1):
            if arr[idx] < arr[left]:
                mid += 1
                arr[idx], arr[mid] = arr[mid], arr[idx]
        arr[mid], arr[left] = arr[left], arr[mid]
        if index == mid:
            return arr[index]
        elif index < mid:
            return _top_k(left, mid - 1)
        else:
            return _top_k(mid + 1, right)

    return _top_k(0, length - 1)


def shortest_path():  # 广度优先遍历,查找无权图最短路径
    class Node:
        def __init__(self, _x, _y, parent=None):
            self.pos = (_x, _y)
            self.parent = parent

    maze = [  # 1代表可达,注意区分与邻接矩阵表示图的区别
        [1, 0, 1, 0, 1, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 0, 1],
        [1, 0, 0, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 1, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 1, 1, 1],
    ]
    m = len(maze)
    n = len(maze[0])
    queue = deque()
    queue.append(Node(0, 0))
    while queue:
        node = queue.popleft()
        x, y = node.pos
        if x == m - 1 and y == n - 1:
            while node:
                print(node.pos)
                node = node.parent
            break
        maze[x][y] = 0
        for i, j in zip([-1, -1, 0, 1, 1, 1, 0, -1], [0, 1, 1, 1, 0, -1, -1, -1]):
            X = x + i
            Y = y + j
            if 0 <= X < m and 0 <= Y < n and maze[X][Y]:
                queue.append(Node(X, Y, node))


def ball_game():
    """
    ABCDE五人互相传球,其中A与B不会互相传球,C只会传给D,E不会传给C,问从A开始第一次传球,经过5次传球后又传回到A有多少种传法
    matrix[i][j]代表经过一次传球i到j所有可能次数
    (matrix@matrix)[i][j]代表经过两次传球i到j所有可能次数
    matrix = np.array([
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 0, 1, 0]
    ])
    method = (matrix @ matrix @ matrix @ matrix @ matrix)[0][0]  # 有向图长度为k路径数问题
    """
    member_a = 4
    queue = deque(((member_a, 0),))
    matrix = [  # # # A B C D E
        0b11010,  # E
        0b11101,  # D
        0b00010,  # C
        0b00111,  # B
        0b00111,  # A
    ]  # 也可以用邻接表实现
    length = len(matrix)
    method = 0
    while queue:
        member, level = queue.popleft()
        if level >= 5:
            if member == member_a:
                method += 1
        else:
            level += 1
            row = matrix[member]
            for idx in range(length):
                if row >> idx & 1:
                    queue.append((idx, level))
    return method


def print_n_queens(columns):
    tmp = [0] * len(columns)
    for column in columns:
        tmp[column] = 1
        print(tmp)
        tmp[column] = 0
    print()


def n_queens(number=8):  # 八皇后问题,低效
    for columns in permutations(range(number)):
        for row in range(1, number):  # 第一行不用检测
            column = columns[row]
            for _row in range(row):
                if row - _row == abs(column - columns[_row]):
                    break
            else:  # 目的是为了跳出2层for循环 ，也可以设置一个bool类型来区分
                continue
            break
        else:
            print_n_queens(columns)


def n_queens_recur(number=8):
    columns = [0] * number  # columns[i]=j意思是第i+1行第j+1列为1

    # stack = []    # 回溯法,此处stack功能等价于rows
    def _solve(level):
        if level < number:
            for i in range(number):
                for j in range(level):
                    if i == columns[j] or level - j == abs(i - columns[j]):  # 注意这里需要对列方向和斜方向做判断
                        break
                else:
                    # stack.append(i)
                    columns[level] = i
                    _solve(level + 1)
                    # stack.pop()
        else:
            print_n_queens(columns)

    _solve(0)


# 程序一行一行地寻找可以放皇后的地方,过程带三个参数row、ld和rd,分别表示在纵列和两个对角线方向的限制条件下这一行的哪些地方不能放
def n_queens_bit(number=8):  # 效率最高
    stack = []

    def _n_queens_bit(row, ld, rd):
        up_limit = (1 << number) - 1
        if row == up_limit:
            for i in stack:
                print([int(j) for j in '{:b}'.format(i).rjust(number, '0')])
            print()
        else:
            position = up_limit & ~(row | ld | rd)  # 每个1比特位代表当前行的对应列可以放皇后
            while position:
                pos = position & -position  # 取最低位1
                position ^= pos  # 最低位1置为0
                stack.append(pos)
                _n_queens_bit(row | pos, (ld | pos) << 1, (rd | pos) >> 1)  # 神来之笔
                stack.pop()

    _n_queens_bit(0, 0, 0)


def hamming_weight_64(number):
    # number = (number & 0x5555555555555555) + ((number >> 1) & 0x5555555555555555)
    # number = (number & 0x3333333333333333) + ((number >> 2) & 0x3333333333333333)
    # number = (number & 0x0F0F0F0F0F0F0F0F) + ((number >> 4) & 0x0F0F0F0F0F0F0F0F)
    # number = (number & 0x00FF00FF00FF00FF) + ((number >> 8) & 0x00FF00FF00FF00FF)
    # number = (number & 0x0000FFFF0000FFFF) + ((number >>16) & 0x0000FFFF0000FFFF)
    # number = (number & 0x00000000FFFFFFFF) + ((number >>32) & 0x00000000FFFFFFFF)
    # return number

    # 进阶版
    number = number - ((number >> 1) & 0x5555555555555555)
    number = (number & 0x3333333333333333) + ((number >> 2) & 0x3333333333333333)
    number = (number + (number >> 4)) & 0x0F0F0F0F0F0F0F0F
    number = number + (number >> 8)
    number = number + (number >> 16)
    number = number + (number >> 32)
    return number & 0x0000007F


# 划分最大无冲突子集问题
def division(conflict_set):
    length = len(conflict_set)
    project_set = deque(range(length))
    pre = length
    group_index = 0
    result = [0] * length
    while project_set:
        index = project_set.popleft()
        if index <= pre:  # 这里必须要有等号,否则当只剩最后一个元素时会陷入死循环
            group_index += 1
            clash = 0
            # clash=set()  # 对应的R=[{1,5},{0,4,5,7,8},{5,6},{4,8},{1,3,6,8},{0,1,2,6},{2,4,5},{1},{1,3,4}]
        if clash >> index & 1:
            # if index in clash:
            project_set.append(index)
        else:
            result[index] = group_index
            clash |= conflict_set[index]
        pre = index
    return result


def transform(exp):  # 中缀表达式转后缀表达式
    operators = {'#': -1, '(': 0, '+': 1, '-': 1, ')': 2, '*': 2, '/': 2, '%': 2}
    stack = ['#']
    suffix = []
    for each in exp:
        if each not in operators:
            suffix.append(each)
        else:
            if each == '(':
                stack.append('(')
            elif each == ')':
                ch = stack.pop()
                while ch != '(':
                    suffix.append(ch)
                    ch = stack.pop()
            else:
                while operators[stack[-1]] >= operators[each]:
                    suffix.append(stack.pop())
                stack.append(each)
    while stack:
        suffix.append(stack.pop())
    return suffix


def evaluation(suffix):  # 计算后缀表达式
    stack = []
    operators = {
        '+': lambda x, y: x + y,
        '-': lambda x, y: x - y,
        '*': lambda x, y: x * y,
        '/': lambda x, y: x / y,
        '%': lambda x, y: x % y,
    }
    for ch in suffix[:-1]:  # 最后一个是结束符#
        if ch not in operators:
            stack.append(float(ch))
        else:
            y = stack.pop()
            x = stack.pop()
            stack.append(operators[ch](x, y))
    print(stack.pop())


if __name__ == "__main__":
    '''
    项目project_set = {0, 1, 2, 3, 4, 5, 6, 7, 8}
    冲突集合conflict_set = {
    (1, 4), (4, 8), (1, 8), (1, 7), (8, 3), (1, 0), 
    (0, 5), (1, 5), (3, 4), (5, 6), (5, 2), (6, 2), (6, 4)
    }
    '''
    c_set = [
        0b000100010,
        0b110110001,
        0b001100000,
        0b100010000,
        0b101001010,
        0b001000111,
        0b000110100,
        0b000000010,
        0b000011010,
    ]
    print(division(c_set))
    print(top_k([2, 1, 5, 6, 3, 2, 8, 1, 9, 10, 5, 2, 7], 6))
    shortest_path()
    print(ball_game())
    print(encode_string('qddddftddjhh'))
    n_queens_bit(4)
