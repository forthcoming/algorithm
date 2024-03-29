# 背包问题(要求背包刚好装满,组合问题)
def knapsack(capacity, weights):  # capacity:背包容量 weight:物品的体积集合
    stack = []
    start = 0
    length = len(weights)
    while True:
        for index in range(start, length):
            weight = weights[index]
            if capacity > weight:
                stack.append(index)
                capacity -= weight
            elif capacity == weight:
                print([weights[_] for _ in stack] + [weight])
        if stack:
            start = stack.pop()
            capacity += weights[start]
            start += 1
        else:
            break


# 组合,背包问题的变形
def combination_knapsack(weights):
    stack = []
    start = 0
    length = len(weights)
    while True:
        for index in range(start, length):
            stack.append(index)
            print([weights[idx] for idx in stack])
        if stack:
            start = stack.pop() + 1
        else:
            break


def combination(arr):
    length = len(arr)
    num = 1 << length
    for i in range(1, num):
        print([arr[j] for j in range(length) if i >> j & 1])


def part_combination(arr, m):  # arr在[left,right]范围内,C(m,n+1)= C(m,n)+C(m-1,n)
    def _part_combination(_m, _left, _right):
        if 0 < _m <= _right - _left + 1:
            stack.append(arr[_left])
            _part_combination(_m - 1, _left + 1, _right)
            stack.pop()
            _part_combination(_m, _left + 1, _right)
        elif _m == 0:
            print(stack)

    stack = []
    left = 0
    right = len(arr) - 1
    _part_combination(m, left, right)


def part_combination_v1(arr, m):  # 与part_combination相比就是把C(m-1,n)展开了
    def _part_combination(_m, _left, _right):
        if _m == 0:  # 注意这里的判断条件
            print(stack)
        else:
            for idx in range(_left, _right - _m + 2):
                for j in range(_left, idx):  # 去重,前提是li中相同元素连在一起(why)
                    if arr[idx] == arr[j]:
                        break
                else:
                    stack.append(arr[idx])
                    _part_combination(_m - 1, idx + 1, _right)
                    stack.pop()

    stack = []
    left = 0
    right = len(arr) - 1
    _part_combination(m, left, right)


# 利用位运算进行部分组合,思想参考next_permutation
def bit_part_combination(arr, m):
    length = len(arr)
    minimum = (1 << m) - 1
    maximum = minimum << (length - m)
    while minimum <= maximum:
        print([arr[i] for i in range(length) if minimum >> i & 1])
        b = minimum & -minimum
        t = minimum + b  # 从右到左将首次升序位置的0置为1,其后的1位置0
        minimum = (minimum ^ t) // b >> 2 | t


def combination_bit(arr):
    for i in range(1, len(arr) + 1):
        bit_part_combination(arr, i)


if __name__ == "__main__":
    # knapsack(10, (1, 8, 4, 3, 5, 2))
    # combination_knapsack((1, 8, 4))
    part_combination_v1([0, 1, 2, 3], 3)
