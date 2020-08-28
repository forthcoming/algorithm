# 去重全排列
def full_permutation(arr,left,end):
    if left<end:
        for idx in range(left,end+1):
            for j in range(left,idx):  # 去重
                if arr[idx] == arr[j]:
                    break
            else:
                arr[left],arr[idx] = arr[idx],arr[left]
                full_permutation(arr,left+1,end)
                arr[idx],arr[left] = arr[left],arr[idx]  # 注意此处要还原
    else:
        print(arr)

                
'''
相邻两个位置ai < ai+1,ai称作该升序的首位
步骤:二找、一交换、一翻转
找到排列中最后(最右)一个升序的首位位置i,x = ai
找到排列中第i位右边最后一个比ai大的位置j,y = aj
交换x,y
prePermutation只需要将Ⅰ逆序并将Ⅱ,Ⅲ处<变为>即可
python中通过from itertools import permutations调用全排列
'''
def nextPermutation(s):  #不去重
    length=len(s)
    arr=list(range(length))  # Ⅰ
    while True:
        print([s[i] for i in arr])
        for i in range(length-2,-1,-1):
            if arr[i]<arr[i+1]:   # Ⅱ
                break
        else:
            break
        for j in range(length-1,i,-1):
            if arr[i]<arr[j]:  # Ⅲ
                arr[i],arr[j]=arr[j],arr[i]
                break
        arr[i+1:]=arr[:i:-1]
    
#当需要排列的对象可比较(如全字符or全数字对象)时,可以通过prePermutation+nextPermutation实现去重全排列permutation
def permutation(s):  #去重
    print(s)
    length=len(s)

    right=list(s)
    while True:
        for i in range(length-2,-1,-1):
            if right[i]<right[i+1]:
                break
        else:
            break
        for j in range(length-1,i,-1):
            if right[i]<right[j]:
                right[i],right[j]=right[j],right[i]
                break
        right[i+1:]=right[:i:-1]
        print(right)

    left=list(s)
    while True:
        for i in range(length-2,-1,-1):
            if left[i]>left[i+1]:
                break
        else:
            break
        for j in range(length-1,i,-1):
            if left[i]>left[j]:
                left[i],left[j]=left[j],left[i]
                break
        left[i+1:]=left[:i:-1]
        print(left)
