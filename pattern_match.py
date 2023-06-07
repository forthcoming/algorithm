def bf(text, pattern, pos=0):
    length_t = len(text)
    length_p = len(pattern)
    j = 0
    while pos + j < length_t and j < length_p:
        if text[pos + j] == pattern[j]:
            j += 1
        else:
            pos += 1
            j = 0
    return pos if j == length_p else -1


def rabin_karp(text, pattern, radix=31):
    length_t = len(text)
    length_p = len(pattern)
    if length_t < length_p:
        return -1
    hash_t = 0
    hash_p = 0
    offset = radix ** length_p
    for i in range(length_p):
        hash_t = hash_t * radix + ord(text[i])  # rolling hash
        hash_p = hash_p * radix + ord(pattern[i])
    if hash_t == hash_p:  # 此处可以做进一步检查,字符串是否相等
        return 0
    for j in range(length_p, length_t):
        hash_t = hash_t * radix - ord(text[j - length_p]) * offset + ord(text[j])
        if hash_t == hash_p:  # 此处可以做进一步检查,字符串是否相等
            return j - length_p + 1
    return -1


def build_next(pattern):
    """
    next[j]代表p[0,j-1]子串最大长度的相同前缀后缀(不含子串本身,应为子串至少要移动一步),s[j]匹配失败时下一步匹配中模式串应该跳到next[j]位置
    人工计算步骤:
    1. 计算p[0,0], p[0,1]... p[0,n]子串的最大长度的相同前缀后缀0, max(1), max(2)... max(n)
    2. next[0]=-1, next[1]=0, next[2]=max(1)...next[n]=max(n-1)

    程序计算步骤:
    1. 当pattern长度大于等于2时,next[0]=-1, next[1]=0,由next[j]推导next[j+1]
    2.
      1. 如果p[next[j]]==p[j],next[j+1]=next[j]+1,如果next[j+1]再大点next[j]一定会变大,所以next[j+1]只能等于next[j]+1
      2. 如果p[next[j]]!=p[j],需要在p[0,next[j]-1]子串寻找,即判断p[next[next[j]]]是否等于p[j]
    """
    length_p = len(pattern)
    _next = [-1] * length_p
    k = -1  # 代表优化前的next[pos_p-1]
    pos_p = 1
    while pos_p < length_p:
        if k == -1 or pattern[pos_p - 1] == pattern[k]:
            k += 1
            # 优化_next数组,直接_next[pos_p]=k也行,数组_next中的数字越小则认为越优化
            k1 = k
            while k1 != -1 and pattern[pos_p] == pattern[k1]:  # pattern[pos_p]代表匹配失败的字符
                k1 = _next[k1]
            _next[pos_p] = k1
            pos_p += 1
        else:
            k = _next[k]
    return _next


def kmp(text, pattern, pos_t=0):  # 时间复杂度O(s+p)
    length_t = len(text)
    length_p = len(pattern)
    _next = build_next(pattern)
    pos_p = 0
    while pos_p < length_p and pos_t < length_t:
        if pos_p == -1 or text[pos_t] == pattern[pos_p]:
            pos_t += 1
            pos_p += 1
        else:
            pos_p = _next[pos_p]
    return pos_t - pos_p if pos_p == length_p else -1


def sunday(text, pattern, pos_t=0):
    length_t = len(text)
    length_p = len(pattern)
    mapping = {each: index for index, each in enumerate(pattern)}  # 重要,后面出现的字符index覆盖前面重复出现的字符index
    pos_p = 0
    while pos_p < length_p and pos_t < length_t:
        if pattern[pos_p] == text[pos_t]:
            pos_t += 1
            pos_p += 1
        else:
            pos_t += length_p - pos_p  # 匹配失败时pos_t是text中参加匹配的最末位字符的下一位字符下标,应为这个字符一定会被比较
            if pos_t >= length_t:
                return -1
            else:
                # 将text和pattern的text[pos_t]字符对齐,然后从头开始遍历,没找到返回-1,从pos_t下一个位置开始遍历
                pos_t -= mapping.get(text[pos_t], -1)
                pos_p = 0
    return pos_t - pos_p if pos_p == length_p else -1


if __name__ == "__main__":
    print(kmp("bbc abcdab abcdabcdabde", "abcdabd"))
