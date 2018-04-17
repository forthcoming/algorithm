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
