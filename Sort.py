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
