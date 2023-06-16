"""
base64是一种基于64个可打印字符来表示二进制数据的表示方法,每6个比特为一个单元,对应某个可打印字符,编码后数据是原先的1.333倍
base32是一种基于32个可打印字符来表示二进制数据的表示方法,每5个比特为一个单元,对应某个可打印字符,编码后数据是原先的1.6倍
baseN不是加密,主要用途是把一些二进制数转成普通字符用于网络传输,另外还有一些系统中只能使用ASCII字符,BaseN就是用来将非ASCII字符数据转换成ASCII字符
url编码一般使用base32,应为其字母不区分大小写
base32要求结果串长度必须是8的倍数,所以目标串必须是5的倍数
"""
import base64

alphabet = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
_b32tab2 = None


def encode_base32(message):  # base64.b32encode()
    length = len(message)
    idx = 0
    buffer = 0
    left = 0
    result = bytearray()
    while left > 5 or idx < length:
        if left < 5:
            buffer = 0b1111111111111111 & ((buffer << 8) | message[idx])  # 最多只有12个有效bit位,b字符串按下标取出时会转换为int类型
            left += 8
            idx += 1
        result.append(alphabet[0b11111 & (buffer >> (left - 5))])
        left -= 5
    if left:  # 防止left还有剩余
        result.append(alphabet[0b11111 & (buffer << (5 - left))])
    pad = len(result) % 8
    if pad:
        result.extend([61] * (8 - pad))  # 长度必须是8的整倍数,不足的用=填充
    return result


def encode_base32_v2(message):
    idx = 0
    length = len(message)
    result = bytearray()
    left = 8
    while idx < length:
        if left >= 5:
            result.append(alphabet[message[idx] >> (left - 5) & 0b00011111])
            left -= 5
        if idx + 1 == length:
            break
        else:
            _idx = (message[idx] << (5 - left) | message[idx + 1] >> (3 + left)) & 0b00011111
            result.append(alphabet[_idx])
            idx += 1
            left += 3
    if left:  # 防止left还有剩余
        result.append(alphabet[message[idx] << (5 - left) & 0b11111])
    pad = len(result) % 8
    if pad:
        result.extend([61] * (8 - pad))  # 长度必须是8的整倍数,不足的用=填充
    return result


def encode_base32_v1(s):  # 参考python3源码base64.b32encode
    global _b32tab2
    # Delay the initialization of the table to not waste memory if the function is never called
    if _b32tab2 is None:
        b32tab = [bytes((i,)) for i in alphabet]
        _b32tab2 = [a + b for a in b32tab for b in b32tab]
        b32tab = None
    leftover = len(s) % 5
    # Pad the last quantum with zero bits if necessary
    if leftover:
        s = s + b'\0' * (5 - leftover)  # Don't use +=  b'\0'对应整数0
    encoded = bytearray()
    from_bytes = int.from_bytes
    for i in range(0, len(s), 5):
        # int.from_bytes(b'ab','big') == 0b0110000101100010; int.from_bytes(b'ab','little') == 0b0110001001100001
        c = from_bytes(s[i: i + 5], 'big')
        encoded += (
                _b32tab2[c >> 30] +  # bits 1 - 10
                _b32tab2[(c >> 20) & 0x3ff] +  # bits 11 - 20
                _b32tab2[(c >> 10) & 0x3ff] +  # bits 21 - 30
                _b32tab2[c & 0x3ff]  # bits 31 - 40
        )
    # Adjust for any leftover partial quanta
    if leftover == 1:
        encoded[-6:] = b'======'
    elif leftover == 2:
        encoded[-4:] = b'===='
    elif leftover == 3:
        encoded[-3:] = b'==='
    elif leftover == 4:
        encoded[-1:] = b'='
    return bytes(encoded)


def decode_base32(message):
    length = len(message)
    left = 0
    idx = 0
    buffer = 0
    result = bytearray()
    while idx < length:
        ch = message[idx]
        if 65 <= ch <= 90:  # [A,Z]
            ch = (ch & 0b11111) - 1
        elif 50 <= ch <= 55:  # [2,7]
            ch -= 24
        else:  # b'='
            break
        idx += 1
        left += 5
        buffer = 0b1111111111111111 & ((buffer << 5) | ch)
        if left >= 8:
            result.append(0b11111111 & (buffer >> (left - 8)))
            left -= 8
    return result  # 不用考虑left还有剩余,应为剩余位都是编码时补的0


if __name__ == "__main__":
    info = b"abcde"
    print(decode_base32(encode_base32_v2(info)))
    print(base64.b32decode(base64.b32encode(info)))
