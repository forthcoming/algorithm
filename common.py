from collections import deque
import random

# Josephus problem
class Node:
    def __init__(self,data,right=None):
        self.data=data
        self.right=right
def josephus(L,num):
    for i in range(10):
        for i in range(num-1):
            L=L.right
        print(L.right.data)
        L.right=L.right.right
L=Node(0)
L.right=Node(1,Node(2,Node(3,Node(4,Node(5,Node(6,Node(7,Node(8,Node(9,L)))))))))
josephus(L,3)

f=lambda:random.randrange(0,8)  # 等概率產生[0-7]
def fun():    # 等概率产生[1-30] 
    total=(f()<<3)+f()-2   # [-2,61]
    if total<=1:        
        return fun()    
    else:        
        return total>>1
    
# 筛选素数
def prime(num):
    a=[1 for i in range(0,num+1)]
    for i in range(2,int(num**.5)+1):
        if a[i]:
            j=i
            while j*i<=num:
                a[i*j]=0
                j+=1
    return [i for i in range(2,num+1) if a[i]]

# 广度优先遍历,查找无权图最短路径
def shortest_path():
    class Node:
        def __init__(self,x,y,left=None):
            self.pos=(x,y)
            self.left=left
    maze=[  # 1代表可达,注意区分与邻接矩阵表示图的区别
        [1,0,1,0,1,1,1,0],   
        [0,1,1,0,0,1,0,1],
        [1,0,0,1,1,0,0,0],
        [0,1,1,0,0,1,1,0],
        [0,1,1,1,0,0,1,0],
        [1,0,0,0,1,1,1,1],
    ]
    m=len(maze)
    n=len(maze[0])
    queue=deque()
    queue.append(Node(0,0))
    while queue:
        node=queue.popleft()
        x,y=node.pos
        if x==m-1 and y==n-1:
            while node:
                print(node.pos)
                node=node.left
            break
        maze[x][y]=0
        for i,j in zip([-1,-1,0,1,1,1,0,-1],[0,1,1,1,0,-1,-1,-1]):
            X=x+i
            Y=y+j
            if 0<=X<m and 0<=Y<n and maze[X][Y]:
                queue.append(Node(X,Y,node))
             
'''
选择数组中最大的k个数
1.构建一个大顶堆
2.构建一个大小为k的小顶堆
3.快排变形
'''
def topK(li,left,right,k,result):  #不包含right,结果存入result
    if 0<k<=right-left:
        index=left
        for i in range(left+1,right):
            if li[i]<li[left]:
                index+=1
                li[i],li[index]=li[index],li[i]
        li[index],li[left]=li[left],li[index]
        if right-index>k:
            topK(li,index+1,right,k,result)
        else:
            result+=li[index:right]
            if right-index<k:
                topK(li,left,index,k-right+index,result)

'''
哈希
数据量n/哈希表长度m=[.65,.85],比值越小效率越高
处理冲突的方法有开放地址法,链地址法(推荐),前者不太适合删除操作,应为删除的元素要做特殊标记
哈希函数的值域必须在表长范围之内，同时希望关键字不同所得哈希函数值也不同
'''
# 数字哈希
Hash=lambda num,m,A=(5**.5-1)/2:int(A*num%1*m)  # 除留取余法,平方取中法(按比特位取中),折叠法

# 字符串哈希
def BKDRHash(string,radix=31):
    # radix 31 131 1313 13131 131313 etc.
    
    string = bytearray(string.encode())   
    hash=0
    for i in string:
        hash=hash*radix+i
    return hash

'''
幂运算问题
如果采用递归slow(x,y)=x*slow(x,y-1)效率会很慢
分治法降低power时间复杂度到logn,效率 x**y = pow > power > slow
'''
def power(x,y):  # y为任意整数
    if not y:
        return 1
    elif y==1:
        return x
    elif y==-1:
        return 1/x
    elif y&1:
        return x*power(x,y-1)
    else:
        return power(x*x,y>>1)

'''
牛顿/二分法求平方根问题(幂级数展开也能求近似值)
# Newton法求f(x)=x**4+x-10=0在[1,2]内的一个实根
x=1  # x也可以是2
for i in range(10):
    # x-=(x**4+x-10)/(4*x**3+1)
    x=(3*x**4+10)/(4*x**3+1)
'''
def sqrt(t,precision=10):
    assert t>0 and type(precision)==int and precision>0
    border=t  # border也可以是2t等
    left=0
    right=t
    for i in range(precision):
        border=.5*(border+t/border)   #牛顿法,收敛速度快
        mid=(left+right)/2  #二分法,收敛速度很慢
        if mid**2>=t:
            right=mid
        else:
            left=mid
    print(f'牛顿法结果:{border}\n二分法结果:{mid}')

def bisect_right(a, x, lo=0, hi=None):
    """
    Return the index where to insert item x in list a, assuming a is sorted.
    The return value i is such that all e in a[:i] have e <= x, and all e in a[i:] have e > x.  
    So if x already appears in the list, a.insert(x) will insert just after the rightmost x already there.
    """
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)>>1
        if x < a[mid]: 
            hi = mid
        else: 
            lo = mid+1
    return lo

# 二分查找,序列必须有序,ASL=(n+1)*log(n+1)/n - 1
def BinarySearch(li,left,right,num):
    while left<=right:
        index=(left+right)>>1
        if li[index]>num:
            right=index-1
        elif li[index]<num:
            left=index+1
        else:
            return index
    return -1

def RecurBinarySearch(li,left,right,num):
    if left<=right:
        index=(left+right)>>1
        if li[index]>num:
            right=index-1
        elif li[index]<num:
            left=index+1
        else:
            return index
        return RecurBinarySearch(li,left,right,num)
    else:
         return -1

# 移位有序数组查找(eg: li=[4,5,6,7,8,9,0,1,2,3])
def search(li,left,right,num):
    if left<=right:
        mid=(left+right)>>1
        if li[left]<=li[mid]:
            if li[left]<=num<=li[mid]:
                return BinarySearch(li,left,mid,num)
            else:
                return search(li,mid+1,right,num)
        else:
            return max(search(li,left,mid,num),BinarySearch(li,mid+1,right,num))
    else:
        return -1

# 求连续子列和的最大值(也可以用分治法,不过时间复杂度是nlogn)
def maxSum(li):
    curSum=MAX=0
    for i in li:
        curSum+=i
        if curSum>MAX:
            MAX=curSum
        elif curSum<0:
            curSum=0
    return MAX

# 寻找和为定值的两个数(towsum([1,3,4,5,6,7,8,9,10,11],12))
def towsum(l,num): #前提是l有序，如果无序，可考虑先线性排序（参照桶排序），或者直接边哈希边判断(Python可以使用set)
    begin,end=0,len(l)-1
    while begin<end:
        total=l[begin]+l[end]
        if total==num:
            print('begin:{},end:{},l[begin]:{},l[end]:{}'.format(begin,end,l[begin],l[end]))
            end-=1
            begin+=1
        elif total>num:
            end-=1
        else:
            begin+=1
