# 排列
def arrangement(s,left,right):
    if left==right:
        print(s)
    else:
        for j in range(left,right+1):
            for k in range(left,j):   #去重
                if s[k]==s[j]:
                    break
            else: 
                s[j],s[left]=s[left],s[j]
                arrangement(s,left+1,right)
                s[j],s[left]=s[left],s[j]   #注意此处要还原
                
'''
相邻两个位置ai < ai+1,ai称作该升序的首位
步骤:二找、一交换、一翻转
找到排列中最后(最右)一个升序的首位位置i,x = ai
找到排列中第i位右边最后一个比ai大的位置j,y = aj
交换x,y
prePermutation只需要将Ⅰ逆序并将Ⅱ,Ⅲ处<变为>即可
'''
def nextPermutation(s):  #不去重
    length=len(s)
    arr=list(range(length))  # Ⅰ
    while True:
        print(''.join(s[i] for i in arr))
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
