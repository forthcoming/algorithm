# 背包问题(要求背包刚好装满,组合问题)
def knapsack(T,weight):  # T:背包容量 weight:物品的体积集合 knapsack(10,(1,8,4,3,5,2))
    stack=[]
    length=len(weight)
    index=0
    while stack or (index<length):
        while T and index<length:
            if T>=weight[index]:
                stack.append(index)
                T-=weight[index]
            index+=1   
        if not T:
            print([weight[idx] for idx in stack])
        index=stack.pop()
        T+=weight[index]
        index+=1

# 组合,背包问题的变形
def combination1(weight): 
    stack=[]
    length=len(weight)
    index=0
    while stack or (index<length):
        while index<length:
            stack.append(index)
            print([weight[idx] for idx in stack])
            index+=1   
        index=stack.pop()+1
        
def combination2(s):
    length=len(s)
    num=1<<length
    for i in range(1,num):
        print([s[j] for j in range(length) if i>>j&1])

# 利用位运算进行部分组合,思想参考nextPermutation
def bi_part_combination(li,m): 
    length=len(li)
    minimum=(1<<m)-1
    maximum=minimum<<(length-m)
    while minimum<=maximum:
        print([li[i] for i in range(length) if minimum>>i&1])
        b=minimum & -minimum
        t=minimum+b
        minimum=(minimum^t)//b>>2|t
