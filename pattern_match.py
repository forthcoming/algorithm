def BF(S,T,pos=0):
    sLength=len(S)
    tLength=len(T)
    j=0
    while pos+j<sLength and j<tLength:
        if S[pos+j]==T[j]:
            j+=1
        else:
            pos+=1
            j=0
    return pos if j==tLength else -1

def KMP(S,T,pos=0):
    sLength=len(S)
    tLength=len(T)
    # 求串T的next数组
    _next=[-1]*tLength
    k=-1
    index=1
    while index<tLength:
        if k==-1 or T[index-1]==T[k]:
            k+=1
            # 优化_next数组,直接_next[index]=k也行,数组_next中的数字越小则认为越优化
            _next[index]=_next[k] if T[index]==T[k] else k
            index+=1
        else:
            k=_next[k]

    print(_next)
    index=0
    while index<tLength and pos<sLength:
        if index==-1 or S[pos]==T[index]:
            pos+=1
            index+=1
        else:
            index=_next[index]
    return pos-index if index==tLength else -1

def Sunday(S,T,pos=0):
    sLength=len(S)
    tLength=len(T)
    hashTable={each:index for index,each in enumerate(T)}  # 重要,后面出现的字符index覆盖前面重复出现的字符index
    index=0
    while index<tLength and pos<sLength:
        if T[index]==S[pos]:
            pos+=1
            index+=1
        else:
            pos+=tLength-index      # 注意这里的位置
            if pos>=sLength:
                return -1
            else:
                pos-=hashTable.get(S[pos],-1)   # 注意这里如果没找到则返回-1
                index=0
    return pos-index if index==tLength else -1

def rollingHash(source,pattern,radix=31):
    sourceLength=len(source)
    patternLength=len(pattern)
    if sourceLength<patternLength:
        return -1
    sourceHash=0
    patternHash=0
    offset=radix**patternLength
    for i in range(patternLength):
        sourceHash=sourceHash*radix+ord(source[i])
        patternHash=patternHash*radix+ord(pattern[i])
    if sourceHash==patternHash:
        return 0
    for j in range(patternLength,sourceLength):
        sourceHash=sourceHash*radix+ord(source[j])-ord(source[j-patternLength])*offset
        if sourceHash==patternHash:  # 此处可以做进一步检查,字符串是否相等
            return j-patternLength+1
    return -1
