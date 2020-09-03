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
        
def combination2(li):
    length=len(li)
    num=1<<length
    for i in range(1,num):
        print([li[j] for j in range(length) if i>>j&1])

def part_combination(li,m,left=0,stack=[]):  # C(m,n+1)= C(m,n)+C(m-1,n)
    n=len(li)
    if m==0:
        print([li[_] for _ in stack])
    elif 0<m<=n-left:
        stack.append(left)
        part_combination(li,m-1,left+1)
        stack.pop()
        part_combination(li,m,left+1)

def part_combination1(li,m,left=0,stack=[]):  # 注意stack的位置,C(m,n),与part_combination相比就是把C(m-1,n)展开了
    n=len(li)
    if m>0:
        for i in range(left,n-m+1):
            for j in range(left,i):  # 去重
                if li[i]==li[j]:
                    break
            else:
                stack.append(i)
                part_combination1(li,m-1,i+1)
                stack.pop()
    else:
        print([li[i] for i in stack])

# 利用位运算进行部分组合,思想参考next_permutation
def bi_part_combination(li,m): 
    length=len(li)
    minimum=(1<<m)-1
    maximum=minimum<<(length-m)
    while minimum<=maximum:
        print([li[i] for i in range(length) if minimum>>i&1])
        b=minimum & -minimum   
        t=minimum+b    # 从右到左将首次升序位置的0置为1,其后的1位置0
        minimum=(minimum^t)//b>>2|t

def combination3(li):
    for i in range(1,len(li)+1):
        bi_part_combination(li,i)
