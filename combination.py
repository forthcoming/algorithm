# 背包问题(要求背包刚好装满,组合问题)
def knapsack(capacity, weight):  # capacity:背包容量 weight:物品的体积集合
    stack = []
    index = 0
    length = len(weight)
    while True:
        while capacity and index < length:
            if capacity >= weight[index]:
                stack.append(index)
                capacity -= weight[index]
            index += 1
        if not capacity:
            print([weight[idx] for idx in stack])
        if stack:
            index = stack.pop()
            capacity += weight[index]
            index += 1
        else:
            break


# 组合,背包问题的变形
def combination1(weight):
    stack = []
    length = len(weight)
    index = 0
    while True:
        while index < length:
            stack.append(index)
            print([weight[idx] for idx in stack])
            index += 1
        if stack:
            index = stack.pop() + 1
        else:
            break


def combination2(li):
    length = len(li)
    num = 1 << length
    for i in range(1, num):
        print([li[j] for j in range(length) if i >> j & 1])


def part_combination(li, m, left, right, stack=[]):  # li在[left,right]范围内,C(m,n+1)= C(m,n)+C(m-1,n)
    if 0 < m <= right - left + 1:
        stack.append(li[left])
        part_combination(li, m - 1, left + 1, right)
        stack.pop()
        part_combination(li, m, left + 1, right)
    elif m == 0:
        print(stack)


def part_combination1(li, m, left, right, stack=[]):  # 与part_combination相比就是把C(m-1,n)展开了
    if m == 0:  # 注意这里的判断条件
        print(stack)
    else:
        for idx in range(left, right - m + 2):
            for j in range(left, idx):  # 去重,前提是li中相同元素连在一起(why)
                if li[idx] == li[j]:
                    break
            else:
                stack.append(li[idx])
                part_combination1(li, m - 1, idx + 1, right)
                stack.pop()


# 利用位运算进行部分组合,思想参考next_permutation
def bit_part_combination(li, m):
    length = len(li)
    minimum = (1 << m) - 1
    maximum = minimum << (length - m)
    while minimum <= maximum:
        print([li[i] for i in range(length) if minimum >> i & 1])
        b = minimum & -minimum
        t = minimum + b  # 从右到左将首次升序位置的0置为1,其后的1位置0
        minimum = (minimum ^ t) // b >> 2 | t


def combination3(li):
    for i in range(1, len(li) + 1):
        bit_part_combination(li, i)


if __name__ == "__main__":
    knapsack(10, (1, 8, 4, 3, 5, 2))
