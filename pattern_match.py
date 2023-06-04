def bf(source, pattern, pos=0):
    source_length = len(source)
    pattern_length = len(pattern)
    j = 0
    while pos + j < source_length and j < pattern_length:
        if source[pos + j] == pattern[j]:
            j += 1
        else:
            pos += 1
            j = 0
    return pos if j == pattern_length else -1


def rolling_hash(source, pattern, radix=31):
    source_length = len(source)
    pattern_length = len(pattern)
    if source_length < pattern_length:
        return -1
    source_hash = 0
    pattern_hash = 0
    offset = radix ** pattern_length
    for i in range(pattern_length):
        source_hash = source_hash * radix + ord(source[i])
        pattern_hash = pattern_hash * radix + ord(pattern[i])
    if source_hash == pattern_hash:  # 此处可以做进一步检查,字符串是否相等
        return 0
    for j in range(pattern_length, source_length):
        source_hash = source_hash * radix - ord(source[j - pattern_length]) * offset + ord(source[j])
        if source_hash == pattern_hash:  # 此处可以做进一步检查,字符串是否相等
            return j - pattern_length + 1
    return -1


def build_next(pattern):
    """
    next[j]代表p[0,j-1]子串最大长度的相同前缀后缀(不含子串本身,应为子串至少要移动一步),s[j]匹配失败时下一步匹配中模式串应该跳到next[j]位置
    计算步骤:
    1. 计算p[0,0], p[0,1]... p[0,n]子串的最大长度的相同前缀后缀0, max(1), max(2)... max(n)
    2. next[0]=-1, next[1]=0, next[2]=max(1)...next[n]=max(n-1)
    """
    pattern_length = len(pattern)
    _next = [-1] * pattern_length
    k = -1
    p_pos = 1
    while p_pos < pattern_length:
        if k == -1 or pattern[p_pos - 1] == pattern[k]:
            k += 1
            # 优化_next数组,直接_next[p_pos]=k也行,数组_next中的数字越小则认为越优化
            if pattern[p_pos] == pattern[k]:
                _next[p_pos] = _next[k]
                # _next[p_pos] = k
            else:
                _next[p_pos] = k
            p_pos += 1
        else:
            k = _next[k]
    return _next


def kmp(source, pattern, s_pos=0):
    source_length = len(source)
    pattern_length = len(pattern)
    _next = build_next(pattern)
    p_pos = 0
    while p_pos < pattern_length and s_pos < source_length:
        if p_pos == -1 or source[s_pos] == pattern[p_pos]:
            s_pos += 1
            p_pos += 1
        else:
            p_pos = _next[p_pos]
    return s_pos - p_pos if p_pos == pattern_length else -1


def sunday(source, pattern, pos=0):
    source_length = len(source)
    pattern_length = len(pattern)
    hash_table = {each: index for index, each in enumerate(pattern)}  # 重要,后面出现的字符index覆盖前面重复出现的字符index
    index = 0
    while index < pattern_length and pos < source_length:
        if pattern[index] == source[pos]:
            pos += 1
            index += 1
        else:
            pos += pattern_length - index  # 注意这里的位置
            if pos >= source_length:
                return -1
            else:
                pos -= hash_table.get(source[pos], -1)  # 注意这里如果没找到则返回-1
                index = 0
    return pos - index if index == pattern_length else -1


if __name__ == "__main__":
    print(kmp("bbc abcdab abcdabcdabde", "abcdabd"))
