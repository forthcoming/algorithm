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
