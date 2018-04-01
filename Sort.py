# 归并排序(稳定排序，时间复杂度永远是nlogn,跟数组的数据无关)
def merge(li,left,mid,right):
    result=[]
    p1=left
    p2=mid+1
    while p1<=mid and p2<=right:
        if li[p1] < li[p2]:
            result.append(li[p1])
            p1 += 1
        else:
            result.append(li[p2])
            p2 += 1
    if p2==right+1:
        p2=right-mid+p1
        li[p2:right+1]=li[p1:mid+1]
    li[left:p2]=result

def MergeSort(li,left,right):  #递归版
    if left<right:
        mid = (left+right)>>1
        MergeSort(li,left,mid)
        MergeSort(li,mid+1,right)
        merge(li,left,mid,right)

# 快速排序(非稳定排序)
from random import randrange
def QuickSort(li,left,right):  # 包含left,right边界
    if left<right:   # 效率最高
        flag=li[left]
        start=left
        end=right
        while start<end:  #不能有等号
            while start<end and li[end]>=flag:  #不能有等号
                end-=1
            li[start]=li[end]
            while start<end and li[start]<=flag:   #不能有等号
                start+=1
            li[end]=li[start] 
        li[start]=flag      #此时start等与end
        QuickSort(li,left,start-1)
        QuickSort(li,start+1,right)
        
    # if left<right:  # 效率一般
    #     index=randrange(left,right+1)  # 防止数组本身基本有序带来的效率损失
    #     li[left],li[index]=li[index],li[left]
    #     mid=left
    #     for i in range(left+1,right+1):
    #         if li[i]<li[left]:
    #             mid+=1
    #             li[i],li[mid]=li[mid],li[i]
    #     li[left],li[mid]=li[mid],li[left]
    #     QuickSort(li,left,mid-1)
    #     QuickSort(li,mid+1,right)
        
    # if left<right:  # 效率较低
    #     key=li[left]
    #     start,end=left+1,right
    #     while start<=end:
    #         if li[start]>key:
    #             li[start],li[end]=li[end],li[start]
    #             end-=1
    #         else:
    #             start+=1     
    #     li[start-1],li[left]=li[left],li[start-1]
    #     QuickSort(li,left,start-2)
    #     QuickSort(li,start,right)
    
# 希尔排序
def ShellSort(li):
    length=len(li)
    step=length>>1
    # while step:
    #     for i in range(step):  #遍历每个分组
    #         for j in range(i+step,length,step):
    #             tmp=li[j]
    #             while j>=step and tmp<li[j-step]:
    #                 li[j]=li[j-step]
    #                 j-=step
    #             li[j]=tmp
    #     step>>=1

    while step:  #效率与上面一样,只不过从不同的方向思考问题
        for i in range(step,length):  #遍历每一个数组元素,然后再跟其对应的分组元素做比较
            index=i-step
            tmp=li[i]
            while index>=0 and li[index]>tmp:
                li[index+step]=li[index]
                index-=step
            li[index+step]=tmp
        step>>=1
        
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
def SelectSort(li):
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
    
