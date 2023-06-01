# 基数排序(时间复杂度为O(d*n),特别适合待排记录数n很大而关键字位数d很小的自然数)
def radix_sort(arr, radix=10):
    num = 1 << radix
    length = len(arr)
    bucket = [None] * length
    count = [0] * num
    get_bit = lambda x, y: x >> (y * radix) & (num - 1)
    bit = 0
    while True:
        for i in arr:
            count[get_bit(i, bit)] += 1  # 有点计数排序的思想
        if count[0] == length:  # 说明已经遍历完所有位
            break
        # for j in range(len(count) - 2, -1, -1):  # 逆序
        #     count[j] += count[j + 1]
        for j in range(1, len(count)):
            count[j] += count[j - 1]
        for k in arr[::-1]:  # 注意这里要逆序遍历
            index = get_bit(k, bit)
            bucket[count[index] - 1] = k
            count[index] -= 1
        arr[:] = bucket  # li=bucket并不会改变外面的li对象
        count = [0] * num
        bit += 1


# MSD-10进制递归版基数排序
def msd_radix_sort(arr, left, right, n=5, radix=10):
    if n and left < right:
        bucket = [None] * (right - left + 1)
        count = [0] * radix
        get_bit = lambda x, y: x // radix ** (y - 1) % radix
        for i in arr[left:right + 1]:
            count[get_bit(i, n)] += 1
        for j in range(1, radix):
            count[j] += count[j - 1]
        for k in arr[left:right + 1]:  # 正序逆序都可以
            index = get_bit(k, n)
            bucket[count[index] - 1] = k
            count[index] -= 1
        arr[left:right + 1] = bucket  # 注意这里要加1
        n -= 1
        for x in range(0, radix - 1):  # 遍历count每一个元素
            msd_radix_sort(arr, left + count[x], left + count[x + 1] - 1, n, radix)  # attention
        msd_radix_sort(arr, left + count[-1], right, n, radix)  # attention


# 桶排序(效率跟基数排序类似,实质是哈希,radix越大,空间复杂度越大,时间复杂度越小,但大到一定限度后时间复杂度会增加,适用于自然数)
def bucket_sort(arr, radix=10):
    lower = (1 << radix) - 1
    bucket = [[] for i in range(lower + 1)]  # 不能用 [[]]*(lower+1)
    bit = 0
    while True:
        for val in arr:
            bucket[val >> (bit * radix) & lower].append(val)  # 很关键,原理是将自然数看成二进制数,然后再按lower个位数划分
        if len(bucket[0]) == len(arr):
            break
        del arr[:]
        for each in bucket:
            # for each in bucket[::-1]:   # 逆序
            arr += each  # 部分each为[]
            each[:] = []  # 清空桶数据
        bit += 1
        '''
        下面2种li赋值方法效率很低
        li[:]=reduce(lambda x,y:x+y,bucket)
        li[:]=sum(bucket,[])
        '''


# 快速排序(非稳定排序)
def quick_sort(arr, left, right):  # 包含left,right边界
    if left < right:  # 效率最高
        flag = arr[left]
        start = left
        end = right
        while start < end:  # 不能有等号
            while start < end and arr[end] >= flag:  # 不能有等号
                end -= 1
            arr[start] = arr[end]
            while start < end and arr[start] <= flag:  # 不能有等号
                start += 1
            arr[end] = arr[start]
        arr[start] = flag  # 此时start等与end
        quick_sort(arr, left, start - 1)
        quick_sort(arr, start + 1, right)

    # if left < right:  # 效率一般
    #     index = randrange(left, right + 1)  # 防止数组本身基本有序带来的效率损失
    #     arr[left], arr[index] = arr[index], arr[left]
    #     mid = left
    #     for i in range(left + 1, right + 1):
    #         if arr[i] < arr[left]:
    #             mid += 1
    #             arr[i], arr[mid] = arr[mid], arr[i]
    #     arr[left], arr[mid] = arr[mid], arr[left]
    #     quick_sort(arr, left, mid - 1)
    #     quick_sort(arr, mid + 1, right)
    #
    # if left < right:  # 效率较低
    #     key = arr[left]
    #     start, end = left + 1, right
    #     while start <= end:
    #         if arr[start] > key:
    #             arr[start], arr[end] = arr[end], arr[start]
    #             end -= 1
    #         else:
    #             start += 1
    #     arr[start - 1], arr[left] = arr[left], arr[start - 1]
    #     quick_sort(arr, left, start - 2)
    #     quick_sort(arr, start, right)


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


# 位排序(仅适用于不重复的自然数)
def bit_sort(arr):
    import math
    bitmap = 0
    for each in arr:
        bitmap |= 1 << each

    while bitmap:  # 适用于数的范围较零散
        t = math.log2(bitmap & -bitmap)  # x&-x返回最低位1
        print(int(t), end=' ')
        bitmap &= bitmap - 1  # x&=x-1清除最低位1

    # index = -1  # 适用于数的范围较稠密
    # while bitmap:
    #     index += 1
    #     if bitmap & 1:
    #         print(index, end=' ')
    #     bitmap >>= 1


# 冒泡排序
def bubble_sort(arr):  # 针对此类[random.randrange(0,1000,3) for i in range(2000)]+list(range(3000))大数基本靠右的效率更高
    last_change = len(arr) - 1
    flag = -1
    while flag != last_change:
        flag = last_change
        for i in range(last_change):
            if arr[i] > arr[i + 1]:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                last_change = i

    # length = len(arr)
    # flag = True
    # for i in range(1, length):  # 控制次数
    #     for j in range(length - i):
    #         if arr[j] > arr[j + 1]:
    #             arr[j], arr[j + 1] = arr[j + 1], arr[j]
    #             flag = False
    #     if flag:
    #         break


# 地精排序
def gnome_sort(arr):
    length = len(arr)
    i = 1
    while i != length:
        if i and arr[i] < arr[i - 1]:
            arr[i], arr[i - 1] = arr[i - 1], arr[i]
            i -= 1
        else:
            i += 1


# 插入排序
def insert_sort(arr):
    length = len(arr)
    for i in range(1, length):
        tmp = arr[i]
        while i and tmp < arr[i - 1]:
            arr[i] = arr[i - 1]  # 避免相邻两项两两交换
            i -= 1
        arr[i] = tmp

    # for i in range(1, length):  # 效率较低
    #     for j in range(i, 0, -1):
    #         if arr[j] < arr[j - 1]:
    #             arr[j], arr[j - 1] = arr[j - 1], arr[j]
    #         else:
    #             break


# 选择排序(非稳定排序)
def select_sort(arr):
    length = len(arr)
    for i in range(1, length):
        index = i - 1
        for j in range(i, length):
            if arr[index] > arr[j]:
                index = j
        arr[index], arr[i - 1] = arr[i - 1], arr[index]

    # left = 0
    # right = len(arr) - 1
    # while left < right:
    #     big = small = left
    #     for i in range(left + 1, right + 1):
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
