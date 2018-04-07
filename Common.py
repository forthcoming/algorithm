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
    hash=0
    for i in string:
        hash=hash*radix+ord(i)
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
    else:
        mid=y>>1
        return power(x,mid)*power(x,y-mid)

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
