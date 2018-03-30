# 位排序(仅适用于不重复的自然数)
def BitSort(li):   
    import math
    bitmap=0
    for each in li:
        bitmap|=1<<each

    while bitmap:  #适用于数的范围较零散
        t=math.log2(bitmap&-bitmap) #x&-x返回最低位1
        print(int(t),end=' ')
        bitmap&=bitmap-1            #x&=x-1清除最低位1

    # index=-1       #适用于数的范围较稠密
    # while bitmap:
    #     index+=1
    #     if bitmap&1:
    #         print(index,end=' ')
    #     bitmap>>=1   
    
# 冒泡排序
def BubbleSort(li):  #针对此类[random.randrange(0,1000,3) for i in range(2000)]+list(range(3000))大数基本靠右的效率更高
    lastChange=len(li)-1
    flag=-1
    while flag!=lastChange:
        flag=lastChange
        for i in range(lastChange):
            if li[i]>li[i+1]:
                li[i],li[i+1]=li[i+1],li[i]
                lastChange=i
   
    # length=len(li)
    # flag=True
    # for i in range(1,length):   #控制次数
    #     for j in range(length-i):
    #         if li[j]>li[j+1]:
    #             li[j],li[j+1]=li[j+1],li[j]
    #             flag=False
    #     if flag:
    #         break

# 地精排序
def GnomeSort(li):
    length=len(li)
    i=1
    while i!=length:
        if i and li[i]<li[i-1]:
            li[i],li[i-1]=li[i-1],li[i]
            i-=1
        else:
            i+=1

# 插入排序
def InsertSort(li):
    length=len(li)
    for i in range(1,length):
        tmp=li[i]
        while i and tmp<li[i-1]:
            li[i]=li[i-1]  # 避免相邻两项两两交换
            i-=1              
        li[i]=tmp
        
    # for i in range(1,length): #效率较低
    #     for j in range(i,0,-1):
    #         if li[j]<li[j-1]:
    #             li[j],li[j-1]=li[j-1],li[j]
    #         else:
    #             break

# 选择排序(非稳定排序)
def selectSort(li):
    length=len(li)
    for i in range(1,length):
        index=i-1
        for j in range(i,length):
            if li[index]>li[j]:
                index=j
        li[index],li[i-1]=li[i-1],li[index]

    # left=0
    # right=len(li)-1
    # while left<right:
    #     big=small=left
    #     for i in range(left+1,right+1):
    #         if li[i]>li[big]:
    #             big=i
    #         elif li[i]<li[small]:
    #             small=i
    #     li[left],li[small]=li[small],li[left]
    #     if big==left:  # 注意判断
    #         big=small
    #     li[right],li[big]=li[big],li[right]
    #     left+=1
    #     right-=1
    
