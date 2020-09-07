from collections import deque
import random,os
import numpy as np


class UUID4:
    __slots__ = ('value')

    def __init__(self):
        value = int.from_bytes(os.urandom(16), byteorder='big')  # 0 <= value < 1<<128, 还有int.to_bytes
        version = 4
        # Set the variant to RFC 4122.
        value &= ~(0xc000 << 48)
        value |= 0x8000 << 48
        # Set the version number.
        value &= ~(0xf000 << 64)
        value |= version << 76
        object.__setattr__(self, 'value', value)

    def __setattr__(self, name, value):
        raise TypeError('UUID objects are immutable')

    def __le__(self, other):
        if isinstance(other, UUID4):
            return self.value <= other.value
        return NotImplemented
    
    def __int__(self):
        return self.value
    
    def __str__(self):
        hex = '%032x' % self.value
        return f'{hex[:8]}-{hex[8:12]}-{hex[12:16]}-{hex[16:20]}-{hex[20:]}'

# ABCDE五人互相传球,其中A与B不会互相传球,C只会传给D,E不会穿给C,问从A开始第一次传球,经过5次传球后又传回到A有多少种传法
class Edge:
    def __init__(self,vertex,right=None):
        self.vertex=vertex
        self.right=right
class Graph:
    def __init__(self):
        self.__vertices={}

    def add(self,come,to):
        if come in self.__vertices:
            self.__vertices[come]=Edge(to,self.__vertices[come])
        else:
            self.__vertices[come]=Edge(to)

    def BFS(self,come,to):
        cnt=0
        queue=deque()
        edge=self.__vertices[come]
        queue.append(edge)
        queue.append('#')
        while cnt!=4:
            edge=queue.popleft()
            if edge=='#':
                cnt+=1
                queue.append('#')
            else:
                while edge:
                    vertex=edge.vertex
                    queue.append(self.__vertices[vertex])
                    edge=edge.right
        method=0
        while queue:
            edge=queue.popleft()
            if edge=='#':
                break
            else:
                while edge:
                    vertex=edge.vertex
                    edge=edge.right
                    if vertex==to:
                        method+=1
        return method
# method 1
graph=Graph()
for edge in [('A','C'),('A','D'),('A','E'),('B','C'),('B','D'),('B','E'),('C','D'),('D','A'),('D','B'),('D','C'),('D','E'),('E','A'),('E','B'),('E','D'),]:
    graph.add(*edge)
graph.BFS('A','A')
# method 2
matrix=np.array([[0,0,1,1,1],[0,0,1,1,1],[0,0,0,1,0],[1,1,1,0,1],[1,1,0,1,0]])
(matrix@matrix@matrix@matrix@matrix)[0][0]   # 有向图长度为k路径数问题
'''
matrix[i][j]代表经过一次传球i到j所有可能次数
(matrix@matrix)[i][j]代表经过两次传球i到j所有可能次数
'''

# 统计阶乘数n末尾0的个数,实质就是统计[1,n]中含多少个因子5
def zeros(n):
    step=5
    cnt=0
    while step<=n:
        cnt+=n//step
        step*=5
    return cnt

# 杨氏矩阵查找
# 在一个m行n列二维数组中,每一行都按照从左到右递增的顺序排序,每一列都按照从上到下递增的顺序排序,请完成一个函数,输入这样的一个二维数组和一个整数,判断数组中是否含有该整数
# 以右上角为例,当右上角大于要查找的数字时排除一行,当右上角大于要查找的数字时排除一列
def young_search(li, x):
    m = len(li) - 1
    n = len(li[0]) - 1
    r = 0
    c = n
    while c >= 0 and r <= m:
        value = li[r, c]
        if value == x:
            return True
        elif value > x:
            c = c - 1
        elif value < x:
            r = r + 1
    return False

# 由随机范围[2,7]得到随机范围[5,12]
import random
f=lambda:random.randrange(2,8)  # 等概率產生[2-7]

def range_5_to_12():  # [5,12]
    # f随机长度是6,range_5_to_12长度是8,f至少需要执行2次才能覆盖[5,12]
    total = (f()-2)*6 + f()-2  # [0,35],两个f()不能合并,必须调用2次
    if 32 <=total<= 35:
        return range_5_to_12()    # 此处可以用for循环代替
    return total//4 + 5

def test(n=1000000):
    avg = 0
    for i in range(n):
        avg+=range_5_to_12()
    print(avg/n)
test()    # 8.5
    
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

# 对于一对正整数a,b,对a只能进行加1,减1,乘2操作,最少对a进行几次操作能得到b
def atob(a,b): # BFS 
    result=deque([a,'#'])
    cnt=0
    visited={a}
    while result:
        data=result.popleft()
        if data=='#':
            cnt+=1
            result.append('#')
        elif data==b:
            return cnt
        else:
            if data-1 not in visited:
                result.append(data-1)
                visited.add(data-1)
            if data+1 not in visited:
                result.append(data+1)
                visited.add(data+1)
            if data<<1 not in visited:
                result.append(data<<1)
                visited.add(data<<1)
                
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

def power_stack(x,y):
    result = 1
    if y<0:
        x=1/x
        y=-y
    x_contribute = x
    while y:
        if y&1:
            result*=x_contribute
        x_contribute*=x_contribute
        y>>=1
    return result

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
            hi = mid       # 此处不能写作hi = mid - 1, 可能会越界
        else: 
            lo = mid+1     # 此处必须写作hi = mid + 1, 应为mid只取了整数部分
    return lo

def bisect_right_v1(a, x):
    lo = 0
    hi = len(a)-1
    while lo<=hi:
        mid = (lo+hi)>>1
        if x<a[mid]:
            hi=mid-1
        else:
            lo=mid+1
    return lo

def bisect_left(a, x, lo=0, hi=None):
    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)>>1
        if a[mid] < x:
            lo = mid+1
        else:
            hi = mid
    return lo

# 二分查找,序列必须有序,ASL=(n+1)*log(n+1)/n - 1
def binary_search(li,left,right,num):
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
def cycle_search(li,target):
    left=0
    right=len(li)-1
    while left<=right:
        mid=(left+right)>>1
        if li[mid]==target:
            return mid
        if li[left]<=li[mid]:
            if li[left]<=target<li[mid]:
                right=mid-1
            else:
                left=mid+1
        else:
            if li[mid]<target<=li[right]:
                left=mid+1
            else:
                right=mid-1
    return -1

# 寻找和为定值的两个数(towsum([1,3,4,5,6,7,8,9,10,11],12))
def tow_sum(l,num): #前提是l有序，如果无序，可考虑先线性排序（参照桶排序），或者直接边哈希边判断(Python可以使用set)
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

from collections import deque
def triangles(n):   # 杨辉三角
    queue=deque([1])
    for k in range(n):
        print(' '*(n-k-1),end='')
        queue.append(0)
        while True:
            s=queue.popleft()
            e=queue[0]
            total = s+e
            queue.append(total)
            print(total,end=' ')
            if not e:
                print()
                break

def triangles_lazy():
    L = [1]
    while True:
        yield L
        L.append(0)
        L = [L[i - 1] + L[i] for i in range(len(L))]

        
# 字符串压缩(一串字母(a~z)组成的字符串,将字符串中连续出席的重复字母进行压缩,'ddddftddjh' => '4dft2djh')
def encryption(string):
    result = ''
    number = 0
    tmp = ''
    for char in string:
        if char==tmp:
            number+=1
        else:
            if number==1:
                result+=tmp
            elif number>1:
                result += '{}{}'.format(number, tmp)
            number = 1
            tmp = char
    if number == 1:
        result += tmp
    elif number > 1:
        result += '{}{}'.format(number, tmp)
    return result


# 奇偶调序/rgb排序
def oddEvenSort(l):
    # i,j=0,len(l)-1
    # while i<j:
    #     if not l[i]&1:
    #         l[i],l[j]=l[j],l[i]
    #         j-=1
    #     else:
    #         i+=1

    i=-1
    for j in range(len(l)-1):
        if l[j]&1:
            i+=1
            l[i],l[j]=l[j],l[i]

def rgb(l):
    index,start,end=0,0,len(l)-1
    while index<end:
        if l[index]=='r':
            l[index],l[start]=l[start],l[index]
            start+=1
            index+=1
        elif l[index]=='b':
            l[index],l[end]=l[end],l[index]
            end-=1
        else:
            index+=1
