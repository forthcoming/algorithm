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
                
def nextPermutations(s):  #不去重
    length=len(s)
    arr=list(range(length))
    while True:
        print(''.join(s[i] for i in arr))
        for i in range(length-2,-1,-1):
            if arr[i]<arr[i+1]:
                break
        else:
            break

        for j in range(length-1,i,-1):
            if arr[i]<arr[j]:
                arr[i],arr[j]=arr[j],arr[i]
                break
        arr[i+1:]=arr[:i:-1]
