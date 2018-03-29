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
    
