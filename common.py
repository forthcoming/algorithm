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

import os
import random
from itertools import permutations
from collections import deque

import numpy as np


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

    递归公式catalan_number(m,n) = catalan_number(m-1,n) + catalan_number(m,n-1) ; catalan_number(m,0) = 1; catalan_number(m,n) = 0,m<n
    通项公式catalan_number(m,n) = C(n,m+n) - C(m+1,m+n)
    h(n) = C(2n,n)/(n+1) = C(2n,n)-C(2n,n-1) = h(n-1)*(4n-2)/(n+1) = h(0)*h(n-1)+h(1)*h(n-2)+...+h(n-1)*h(0)
    利用翻折思想,把第一个不符合要求的地方后面的排列互换,就能得到一个新的排列,且该排列跟原先不符合的排列一一对应

    类似问题:错排
    n封信装入不同信封,全装错的个数D(n) = (n-1)[D(n-1) + D(n-2)],思想是寻找子结构,其中D(1)=0, D(2)=1
    """
    # result = [[0] * (n+1) for _ in range(m+1)]
    # for i in range(m+1):
    #     for j in range(n+1):
    #         if j==0:
    #             result[i][j] = 1
    #         elif i>=j:
    #             result[i][j] = result[i-1][j] + result[i][j-1]
    # return result[m][n]

    result = [0] * (n + 1)
    result[0] = 1
    for i in range(1, m + 1):
        for j in range(n + 1):
            if j == 0:
                result[j] = 1
            elif i >= j:
                result[j] += result[j - 1]
            else:
                result[j] = 0
    return result[-1]


# ABCDE五人互相传球,其中A与B不会互相传球,C只会传给D,E不会穿给C,问从A开始第一次传球,经过5次传球后又传回到A有多少种传法
def BFS_search():  # 也可以用邻接表实现
    method = 0
    queue = deque([4, -1])  # 4代表A,-1代表第一层
    matrix = [
        0b11010,
        0b11101,
        0b00010,
        0b00111,
        0b00111,
    ]
    length = len(matrix)
    while True:
        member = queue.popleft()
        if member <= -5:
            break
        elif member < 0:
            queue.append(member - 1)
        else:
            row = matrix[member]
            for idx in range(length):
                if row >> idx & 1:
                    queue.append(idx)
    while queue:
        if queue.pop() == 4:
            method += 1
    return method


def BFS_search():
    method = 0
    queue = deque([0, -1])  # 0代表A,-1代表第一层
    matrix = [
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 0],
        [1, 1, 1, 0, 1],
        [1, 1, 0, 1, 0],
    ]
    length = len(matrix)
    while True:
        member = queue.popleft()
        if member <= -5:
            break
        elif member < 0:
            queue.append(member - 1)
        else:
            row = matrix[member]
            for idx in range(length):
                if row[idx]:
                    queue.append(idx)
    while queue:
        if queue.pop() == 0:
            method += 1
    return method


# matrix[i][j]代表经过一次传球i到j所有可能次数
# (matrix@matrix)[i][j]代表经过两次传球i到j所有可能次数
matrix = np.array([[0, 0, 1, 1, 1], [0, 0, 1, 1, 1], [0, 0, 0, 1, 0], [1, 1, 1, 0, 1], [1, 1, 0, 1, 0]])
_ = (matrix @ matrix @ matrix @ matrix @ matrix)[0][0]  # 有向图长度为k路径数问题


# 判断入栈出栈序列是否合法
def push_pop(push, out):
    stack = []
    push_idx = out_idx = 0
    while push_idx < len(push) and out_idx < len(out):
        item = push[push_idx]
        if item == out[out_idx]:
            out_idx += 1
            push_idx += 1
        elif stack and stack[-1] == out[out_idx]:
            stack.pop()
            out_idx += 1
        else:
            stack.append(item)
            push_idx += 1

    while out_idx < len(out):
        if not stack or stack[-1] != out[out_idx]:
            return False
        out_idx += 1
        stack.pop()
    return True


def push_pop_v1(push, out):
    stack = []
    j = 0
    for i in out:
        if stack and stack[-1] == i:
            stack.pop()
            continue
        while j < len(push):
            j += 1
            if push[j - 1] == i:
                break
            else:
                stack.append(push[j - 1])
        else:
            return False
    return True  # 也可以直接return not stack


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


# 八皇后问题
# 程序一行一行地寻找可以放皇后的地方,过程带三个参数row、ld和rd,分别表示在纵列和两个对角线方向的限制条件下这一行的哪些地方不能放
def eight_queen(num=8, stack=[], row=0, ld=0, rd=0):  # 效率最高
    up_limit = (1 << num) - 1
    if row == up_limit:
        for i in stack:
            print([int(j) for j in '{:b}'.format(i).rjust(num, '0')])
        print()
    else:
        position = up_limit & ~(row | ld | rd)  # 每个1比特位代表当前行的对应列可以放皇后
        while position:
            pos = position & -position  # 取最低位1
            position ^= pos  # 最低位1置为0
            stack.append(pos)
            eight_queen(num, stack, row | pos, (ld | pos) << 1, (rd | pos) >> 1)  # 神来之笔
            stack.pop()


def eight_queen_v1(number=8):
    _ = [0] * number

    # stack = []    # 回溯法
    def _solve(level):
        if level < number:
            for i in range(number):
                for j in range(level):
                    if i == _[j] or level - j == abs(i - _[j]):  # 注意这里需要对列方向和斜方向做判断
                        break
                else:
                    # stack.append(i)
                    _[level] = i
                    _solve(level + 1)
                    # stack.pop()
        else:
            tmp = [0] * number
            for k in _:
                tmp[k] = 1
                print(tmp)
                tmp[k] = 0
            print()

    _solve(0)


def eight_queen_v2(number=8):  # 低效
    for _ in permutations(range(number)):
        for column in range(1, number):  # 第一行不用检测
            row = _[column]
            for _column in range(column):
                if column - _column == abs(row - _[_column]):
                    break
            else:  # 目的是为了跳出2层for循环 ，也可以设置一个bool类型来区分
                continue
            break
        else:
            tmp = [0] * number
            for k in _:
                tmp[k] = 1
                print(tmp)
                tmp[k] = 0
            print()


class UUID4:
    __slots__ = 'value'

    def __init__(self):
        value = int.from_bytes(os.urandom(16), byteorder='big')  # 0 <= value < 1<<128, 还有int.to_bytes
        version = 4
        # Set the variant to RFC 4122.
        value &= ~(0xc000 << 48)
        value |= 0x8000 << 48
        # Set the version number.
        value &= ~(0xf000 << 64)
        value |= version << 76
        object.__setattr__(self, 'value', value)

    def __setattr__(self, name, value):
        raise TypeError('UUID objects are immutable')

    def __le__(self, other):
        if isinstance(other, UUID4):
            return self.value <= other.value
        return NotImplemented

    def __int__(self):
        return self.value

    def __str__(self):
        hexadecimal = '%032x' % self.value
        return f'{hexadecimal[:8]}-{hexadecimal[8:12]}-{hexadecimal[12:16]}-{hexadecimal[16:20]}-{hexadecimal[20:]}'


# 广度优先遍历,查找无权图最短路径
def shortest_path():
    class Node:
        def __init__(self, x, y, left=None):
            self.pos = (x, y)
            self.left = left

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
                node = node.left
            break
        maze[x][y] = 0
        for i, j in zip([-1, -1, 0, 1, 1, 1, 0, -1], [0, 1, 1, 1, 0, -1, -1, -1]):
            X = x + i
            Y = y + j
            if 0 <= X < m and 0 <= Y < n and maze[X][Y]:
                queue.append(Node(X, Y, node))


'''
选择数组中最大的k个数
1.构建一个大顶堆
2.构建一个大小为k的小顶堆
3.快排变形
'''


def topK(li, left, right, k, result):  # 不包含right,结果存入result
    if 0 < k <= right - left:
        index = left
        for i in range(left + 1, right):
            if li[i] < li[left]:
                index += 1
                li[i], li[index] = li[index], li[i]
        li[index], li[left] = li[left], li[index]
        if right - index > k:
            topK(li, index + 1, right, k, result)
        else:
            result += li[index:right]
            if right - index < k:
                topK(li, left, index, k - right + index, result)


'''
牛顿/二分法求平方根问题(幂级数展开也能求近似值)
# Newton法求f(x)=x**4+x-10=0在[1,2]内的一个实根
x=1  # x也可以是2
for i in range(10):
    x=(3*x**4+10)/(4*x**3+1)
'''


def sqrt(t, precision=20):
    assert t > 0 and type(precision) == int and precision > 0
    border = t  # border也可以是2t等
    for i in range(precision):
        border = .5 * (border + t / border)  # 牛顿法,收敛速度快,优于二分法
    print(border)


def bisect_right(a, x, lo=0, hi=None):
    """
    Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e <= x, and all e in a[i:] have e > x.  
    So if x already appears in the list, a.insert(x) will insert just after the rightmost x already there.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) >> 1
        if x < a[mid]:
            hi = mid  # 此处不能写作hi = mid - 1, 可能会越界
        else:
            lo = mid + 1  # 此处必须写作hi = mid + 1, 应为mid只取了整数部分
    return lo


def bisect_right_v1(a, x):
    lo = 0
    hi = len(a) - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        if x < a[mid]:
            hi = mid - 1
        else:
            lo = mid + 1
    return lo


def bisect_left(a, x, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) >> 1
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


# 字符串压缩(一串字母(a~z)组成的字符串,将字符串中连续出席的重复字母进行压缩,'ddddftddjh' => '4dft2djh')
def encryption(string):
    result = ''
    number = 0
    tmp = ''
    for char in string:
        if char == tmp:
            number += 1
        else:
            if number == 1:
                result += tmp
            elif number > 1:
                result += '{}{}'.format(number, tmp)
            number = 1
            tmp = char
    if number == 1:
        result += tmp
    elif number > 1:
        result += '{}{}'.format(number, tmp)
    return result


# 奇偶调序/rgb排序
def odd_even_sort(l):
    # i,j=0,len(l)-1
    # while i<j:
    #    if not l[i]&1:
    #        l[i],l[j]=l[j],l[i]
    #        j-=1
    #    else:
    #        i+=1

    i = -1
    for j in range(len(l) - 1):
        if l[j] & 1:
            i += 1
            l[i], l[j] = l[j], l[i]


def rgb(l):
    index, start, end = 0, 0, len(l) - 1
    while index < end:
        if l[index] == 'r':
            l[index], l[start] = l[start], l[index]
            start += 1
            index += 1
        elif l[index] == 'b':
            l[index], l[end] = l[end], l[index]
            end -= 1
        else:
            index += 1


# 蛇形填数
# def python(n=4):
#     RIGHT,DOWN,LEFT,UP=0,1,2,3
#     direct=RIGHT
#     count=1
#     x=0
#     y=-1
#     lists=[[0]*n for _ in range(n)]
#     for i in range(2*n-1,0,-1):
#         for _ in range((i+1)>>1,0,-1):
#             if direct==RIGHT:
#                 y+=1
#             elif direct==DOWN:
#                 x+=1
#             elif direct==LEFT:
#                 y-=1
#             else:
#                 x-=1
#             lists[x][y]=count
#             count+=1
#         direct=(direct+1)&3
#     for i in lists:
#         print(i)

def python(n=4):  # 更常规
    count = 1
    x = 0
    y = 0
    lists = [[0] * n for _ in range(n)]
    while count <= n ** 2:
        while y < n and not lists[x][y]:
            lists[x][y] = count
            y += 1
            count += 1
        y -= 1
        x += 1
        while x < n and not lists[x][y]:
            lists[x][y] = count
            x += 1
            count += 1
        x -= 1
        y -= 1
        while y >= 0 and not lists[x][y]:
            lists[x][y] = count
            y -= 1
            count += 1
        y += 1
        x -= 1
        while x >= 0 and not lists[x][y]:
            lists[x][y] = count
            x -= 1
            count += 1
        x += 1
        y += 1
    for i in lists:
        print(i)


# 已知正整数a,判断a是否为2的n次方 a&(a-1)或者a-(a&-a)是否等于0
# ^运算符满足交换律, 结合律, x^y^x=y x^x=0 x^0=x


# 划分最大无冲突子集问题
def division(R):
    length = len(R)
    A = deque(range(length))
    pre = length
    group_index = 0
    result = [0] * length
    while A:
        index = A.popleft()
        if index <= pre:  # 这里必须要有等号,否则当只剩最后一个元素时会陷入死循环
            group_index += 1
            clash = 0
            # clash=set()  # 对应的R=[{1,5},{0,4,5,7,8},{5,6},{4,8},{1,3,6,8},{0,1,2,6},{2,4,5},{1},{1,3,4}]
        if clash >> index & 1:
            # if index in clash:
            A.append(index)
        else:
            result[index] = group_index
            clash |= R[index]
        pre = index
    return result


'''
项目A = {0, 1, 2, 3, 4, 5, 6, 7, 8}
冲突集合R = { (1, 4), (4, 8), (1, 8), (1, 7), (8, 3), (1, 0), (0, 5), (1, 5), (3, 4), (5, 6), (5, 2), (6, 2), (6, 4)
'''
R = [
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
print(division(R))


# 中缀表达式转后缀表达式
def transform(exp):
    operators = {'#': -1, '(': 0, '+': 1, '-': 1, ')': 2, '*': 2, '/': 2, '%': 2}
    precede = lambda a, b: operators[a] >= operators[b]
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
                while precede(stack[-1], each):
                    suffix.append(stack.pop())
                stack.append(each)
    while stack:
        suffix.append(stack.pop())
    return suffix


# 计算后缀表达式
def evaluation(suffix):
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
