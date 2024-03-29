import random


def odd_even_sort(arr):  # 奇偶排序/rgb排序
    i, j = 0, len(arr) - 1
    while i < j:
        if arr[i] & 1:
            i += 1
        else:
            arr[i], arr[j] = arr[j], arr[i]
            j -= 1

    # i = -1
    # for j in range(len(arr)):
    #     if arr[j] & 1:
    #         i += 1
    #         arr[i], arr[j] = arr[j], arr[i]


def rgb_sort(arr):
    index, start, end = 0, 0, len(arr) - 1
    while index < end:
        if arr[index] == 'r':
            arr[index], arr[start] = arr[start], arr[index]
            start += 1
            index += 1
        elif arr[index] == 'b':
            arr[index], arr[end] = arr[end], arr[index]
            end -= 1
        else:
            index += 1


def select_sort(arr):  # 选择排序(非稳定排序)
    length = len(arr)
    for i in range(length - 1):
        min_index = i
        for j in range(i + 1, length):
            if arr[min_index] > arr[j]:
                min_index = j
        arr[min_index], arr[i] = arr[i], arr[min_index]

    # left = 0
    # right = len(arr) - 1
    # while left < right:
    #     small = big = left
    #     for i in range(left + 1, right+1):
    #         if arr[i] > arr[big]:
    #             big = i
    #         elif arr[i] < arr[small]:
    #             small = i
    #     arr[left], arr[small] = arr[small], arr[left]
    #     if big == left:  # 注意判断
    #         big = small
    #     arr[right], arr[big] = arr[big], arr[right]
    #     left += 1
    #     right -= 1


# 冒泡排序
def bubble_sort(arr):  # 针对此类[random.randrange(0,1000,3) for i in range(2000)]+list(range(3000))大数基本靠右的效率更高
    last_change = len(arr) - 1
    flag = True
    while flag:
        flag = False
        for i in range(last_change):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                last_change = i
                flag = True


# 插入排序
def insert_sort(arr):
    for i in range(1, len(arr)):
        tmp = arr[i]
        while i and tmp < arr[i - 1]:
            arr[i] = arr[i - 1]  # 避免相邻两项两两交换
            i -= 1
        arr[i] = tmp


# 希尔排序
def shell_sort(arr):
    length = len(arr)
    step = length >> 1
    # while step:
    #     for i in range(step):  # 遍历每个分组
    #         for j in range(i + step, length, step):
    #             tmp = arr[j]
    #             while j >= step and tmp < arr[j - step]:
    #                 arr[j] = arr[j - step]
    #                 j -= step
    #             arr[j] = tmp
    #     step >>= 1

    while step:  # 效率与上面一样,只不过从不同的方向思考问题
        for i in range(step, length):  # 遍历每一个数组元素,然后再跟其对应的分组元素做比较
            index = i - step
            tmp = arr[i]
            while index >= 0 and arr[index] > tmp:
                arr[index + step] = arr[index]
                index -= step
            arr[index + step] = tmp
        step >>= 1


# 快速排序(非稳定排序,时间复杂度nlogn,注意如何计算)
def quick_sort(arr, left, right):  # 包含left,right边界
    if left < right:
        _ = random.randrange(left, right + 1)  # 防止数组本身基本有序带来的效率损失
        arr[left], arr[_] = arr[_], arr[left]
        mid_value = arr[left]
        _left = left
        _right = right
        while _left < _right:  # 不能有等号
            while _left < _right:  # 不能有等号
                if arr[_right] >= mid_value:  # 应为一开始保存了arr[left],所以只能先从右至左遍历
                    _right -= 1
                else:
                    arr[_left] = arr[_right]
                    break
            while _left < _right:  # 不能有等号
                if arr[_left] <= mid_value:
                    _left += 1
                else:
                    arr[_right] = arr[_left]
                    break
        arr[_left] = mid_value  # 此时_left = _right
        quick_sort(arr, left, _left - 1)
        quick_sort(arr, _left + 1, right)

    # if left < right:  # 效率一般
    #     mid = left
    #     for i in range(left + 1, right + 1):
    #         if arr[i] < arr[left]:
    #             mid += 1
    #             arr[i], arr[mid] = arr[mid], arr[i]
    #     arr[left], arr[mid] = arr[mid], arr[left]
    #     quick_sort(arr, left, mid - 1)
    #     quick_sort(arr, mid + 1, right)

    # if left < right:  # 效率一般
    #     _left, _right = left + 1, right
    #     while _left <= _right:
    #         if arr[_left] > arr[left]:
    #             arr[_left], arr[_right] = arr[_right], arr[_left]
    #             _right -= 1
    #         else:
    #             _left += 1
    #     # 此时 _left = _right + 1
    #     arr[_right], arr[left] = arr[left], arr[_right]
    #     quick_sort(arr, left, _right - 1)
    #     quick_sort(arr, _right + 1, right)


def quick_sort_stack(arr):
    args = [(0, len(arr) - 1)]
    while args:
        left, right = args.pop()
        if left < right:
            mid = left
            for index in range(left + 1, right + 1):
                if arr[index] < arr[left]:
                    mid += 1
                    arr[index], arr[mid] = arr[mid], arr[index]
            arr[left], arr[mid] = arr[mid], arr[left]
            args.append((mid + 1, right))
            args.append((left, mid - 1))


def get_pos(natural_number, pos, radix):
    return (natural_number >> (pos * radix)) & ((1 << radix) - 1)


# 桶排序(跟基数排序类似,radix越大,空间复杂度越大,时间复杂度越小,但大到一定限度后时间复杂度会增加,适用于自然数)
def bucket_sort(arr: list[int], radix: int = 10):
    bucket: list[list[int]] = [[] for _ in range(1 << radix)]  # 不能用 [[]]*(1 << radix)
    bit = 0
    while True:
        for element in arr:
            index = get_pos(element, bit, radix)
            bucket[index].append(element)  # 很关键,原理是将自然数看成二进制数,然后再按2**radix个位数划分
        if len(bucket[0]) == len(arr):
            break
        bit += 1

        arr = []
        # for each in bucket[::-1]:   # 逆序
        for each in bucket:
            arr += each  # 部分each为[]
            each[:] = []  # 清空桶数据

        # arr[:] = reduce(lambda x, y: x + y, bucket)
        # arr[:] = sum(bucket, [])


# 基数排序(时间复杂度为O(d*n),特别适合待排记录数n很大而关键字位数d很小的自然数)
def radix_sort(arr, radix=10):
    num = 1 << radix
    length = len(arr)
    bucket = [None] * length
    count = [0] * num
    bit = 0
    while True:
        for element in arr:
            count[get_pos(element, bit, radix)] += 1  # 有点计数排序的思想
        if count[0] == length:  # 说明已经遍历完所有位
            break
        # for j in range(num - 2, -1, -1):  # 逆序
        #     count[j] += count[j + 1]
        for j in range(1, num):
            count[j] += count[j - 1]
        for element in arr[::-1]:  # 注意这里要逆序遍历
            index = get_pos(element, bit, radix)
            bucket[count[index] - 1] = element
            count[index] -= 1
        arr[:] = bucket  # arr=bucket并不会改变外面的arr对象
        count = [0] * num
        bit += 1


# MSD递归版基数排序
def msd_radix_sort(arr, left, right, n=5, radix=10):
    if n and left < right:
        bucket = [None] * (right - left + 1)
        count = [0] * radix
        get_pos = lambda x, y: x // radix ** (y - 1) % radix
        for i in arr[left:right + 1]:
            count[get_pos(i, n)] += 1
        for j in range(1, radix):
            count[j] += count[j - 1]
        for k in arr[left:right + 1]:  # 正序逆序都可以
            index = get_pos(k, n)
            bucket[count[index] - 1] = k
            count[index] -= 1
        arr[left:right + 1] = bucket  # 注意这里要加1
        n -= 1
        for idx in range(0, radix - 1):  # 遍历count每一个元素
            msd_radix_sort(arr, left + count[idx], left + count[idx + 1] - 1, n, radix)  # attention,前提是最高位已经在正确位置
        msd_radix_sort(arr, left + count[-1], right, n, radix)  # attention,前提是最高位已经在正确位置


class MergeSort:
    # 归并排序(稳定排序,时间复杂度永远是nlogn,跟数组的数据无关)
    def __init__(self, arr):
        self.arr = arr  # 待排序数组
        self.inversion_number = 0  # 逆序数,针对recursive_sort方法有效

    def merge(self, left, mid, right):  # 包含[left,mid],[mid+1,right]边界
        result = []
        p1 = left
        p2 = mid + 1
        while p1 <= mid and p2 <= right:
            if self.arr[p1] > self.arr[p2]:
                result.append(self.arr[p2])
                p2 += 1
                self.inversion_number += mid - p1 + 1  # 逆序数统计不管两数相等的情况
            else:
                result.append(self.arr[p1])
                p1 += 1
        if p1 <= mid:
            p2 = right - mid + p1
            self.arr[p2:right + 1] = self.arr[p1:mid + 1]
        self.arr[left:p2] = result

    def sort(self, left, right):  # 递归版归并排序,包含left,right边界
        if left < right:
            mid = (left + right) >> 1
            self.sort(left, mid)
            self.sort(mid + 1, right)
            self.merge(left, mid, right)

    def iter_sort(self):  # 迭代版归并排序
        length = len(self.arr)
        step = 1
        while step < length:
            step <<= 1
            for left in range(0, length, step):  # 以左边界遍历
                right = left + step - 1
                mid = (left + right) >> 1
                if mid >= length:  # 注意
                    mid = length - 1
                if right >= length:  # 注意
                    right = length - 1
                self.merge(left, mid, right)

        # length = len(self.arr)
        # init_mid = 0
        # while init_mid < length - 1:
        #     step = (init_mid + 1) << 1
        #     for mid in range(init_mid, length, step):   # 以中间位置遍历
        #         left = mid - (step >> 1) + 1
        #         right = mid + (step >> 1)  # right=left+step-1
        #         if right >= length:
        #             right = length - 1
        #         self.merge(left, mid, right)
        #     init_mid = (init_mid << 1) + 1

    def reverse(self, left, right):  # [::-1] or list.reverse
        while left < right:
            self.arr[left], self.arr[right] = self.arr[right], self.arr[left]
            left += 1
            right -= 1

    def inplace_merge(self, left, mid, right):  # 包含[left,mid],[mid+1,right]边界,效率低于merge,没更新inversion_number
        mid += 1
        while left < mid <= right:
            p = mid
            while left < mid and self.arr[left] <= self.arr[mid]:
                left += 1
            while mid <= right and self.arr[mid] <= self.arr[left]:
                mid += 1
            self.reverse(left, p - 1)
            self.reverse(p, mid - 1)
            self.reverse(left, mid - 1)
            left += mid - p


if __name__ == "__main__":
    array = [*range(11)]
    random.shuffle(array)
    print(array)  # [0, 8, 4, 5, 7, 6, 3, 2, 1, 9]
    res = MergeSort(array)
    res.iter_sort()
    print(array, res.inversion_number)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 23

    random.shuffle(array)
    quick_sort(array, 0, len(array) - 1)
    print(array)
