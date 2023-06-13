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


# 背包问题(要求背包刚好装满,组合问题)
def knapsack(T, weight):  # T:背包容量 weight:物品的体积集合 knapsack(10,(1,8,4,3,5,2))
    stack = []
    index = 0
    length = len(weight)
    while True:
        while T and index < length:
            if T >= weight[index]:
                stack.append(index)
                T -= weight[index]
            index += 1
        if not T:
            print([weight[idx] for idx in stack])
        if stack:
            index = stack.pop()
            T += weight[index]
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
