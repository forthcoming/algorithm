# 基数排序(时间复杂度为O(d*n),特别适合待排记录数n很大而关键字位数d很小的自然数)
def RadixSort(li,radix=10):
    num=1<<radix
    length=len(li)
    bucket=[None]*length
    count=[0]*num
    getBit=lambda x,y:x>>(y*radix)&(num-1)
    bit=0
    while True:
        for i in li:
            count[getBit(i,bit)]+=1  #有点计数排序的思想
        if count[0]==length:   #说明已经遍历完所有位
            break
        #for j in range(len(count)-2,-1,-1):  #逆序
            #count[j]+=count[j+1]
        for j in range(1,len(count)):
            count[j]+=count[j-1]
        for k in li[::-1]:  #注意这里要逆序遍历
            index=getBit(k,bit)
            bucket[count[index]-1]=k
            count[index]-=1
        li[:]=bucket   #li=bucket并不会改变外面的li对象
        count=[0]*num
        bit+=1
        
# MSD-10进制递归版基数排序
def MSDRadixSort(li,left,right,N=5,radix=10):
    if N and left<right:
        bucket=[None]*(right-left+1)
        count=[0]*radix
        getBit=lambda x,y:x//radix**(y-1)%radix
        for i in range(left,right+1):
            count[getBit(li[i],N)]+=1
        for j in range(1,radix):
            count[j]+=count[j-1]
        for k in range(left,right+1):  # 正序逆序都可以
            index=getBit(li[k],N)
            bucket[count[index]-1]=li[k]
            count[index]-=1
        li[left:right+1]=bucket  #注意这里要加1
        N-=1
        for x in range(0,radix-1):  #遍历count每一个元素
            MSDRadixSort(li,left+count[x],left+count[x+1]-1,N,radix)  # attention
        MSDRadixSort(li,left+count[-1],right,N,radix)   # attention
        
# 桶排序(效率跟基数排序类似,实质是哈希,radix越大,空间复杂度越大,时间复杂度越小,但大到一定限度后时间复杂度会增加,适用于自然数)
def BucketSort(li,radix=10):
    lower=(1<<radix)-1
    bucket = [[] for i in range(lower+1)] # 不能用 [[]]*(lower+1)
    bit=0
    while True:
        for val in li:
            bucket[val>>(bit*radix)&lower].append(val) # 很关键,原理是将自然数看成二进制数,然后再按lower个位数划分
        if len(bucket[0])==len(li):
            break
        del li[:]
        for each in bucket:
        # for each in bucket[::-1]:   # 逆序
            li+=each  # 部分each为[]
            each[:]=[] # 清空桶数据
        bit+=1
        '''
        下面2种li赋值方法效率很低
        li[:]=reduce(lambda x,y:x+y,bucket)
        li[:]=sum(bucket,[])
        '''
        
# 归并排序(稳定排序,时间复杂度永远是nlogn,跟数组的数据无关)
def reverse(li,left,right): #[::-1] or list.reverse
    while left<right:
        li[left],li[right]=li[right],li[left]
        left+=1
        right-=1

def InPlaceMerge(li,left,mid,right): # 包含[left,mid],[mid+1,right]边界,效率低于merge
    mid+=1
    while left<mid and mid<=right:
        p=mid
        while left<mid and li[left]<=li[mid]:
            left+=1
        while mid<=right and li[mid]<=li[left]:
            mid+=1
        reverse(li,left,p-1)    
        reverse(li,p,mid-1)    
        reverse(li,left,mid-1)  
        left+=mid-p
        
def merge(li,left,mid,right): # 包含[left,mid],[mid+1,right]边界
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

def RecurMergeSort(li,left,right):  #递归版归并排序,包含left,right边界
    if left<right:
        mid = (left+right)>>1
        MergeSort(li,left,mid)
        MergeSort(li,mid+1,right)
        merge(li,left,mid,right)
    
def IterMergeSort(li):   #迭代版归并排序
    length=len(li)
    initmid=0
    while initmid<length-1:
        step=(initmid+1)<<1
        for mid in range(initmid,length,step):
            left=mid-(step>>1)+1
            right=mid+(step>>1)  # right=left+step-1            
            if right>=length:
                right=length-1
            merge(li,left,mid,right)
        initmid=(initmid<<1)+1
        
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
    
