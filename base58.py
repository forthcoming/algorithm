'''
Implementations of Base58 and Base58Check endcodings that are compatible with the bitcoin network.
经过Base58编码的数据为原始的数据长度的8/log(2,58)倍,稍稍多于Base64的8/6=1.33倍
'''

from hashlib import sha256
import binascii

alphabet = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
mapping = [
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1,-1,-1,-1,-1,-1,-1,-1, -1,-1,-1,-1,-1,-1,-1,-1,
    -1, 0, 1, 2, 3, 4, 5, 6,  7, 8,-1,-1,-1,-1,-1,-1,
    -1, 9,10,11,12,13,14,15, 16,-1,17,18,19,20,21,-1,
    22,23,24,25,26,27,28,29, 30,31,32,-1,-1,-1,-1,-1,
    -1,33,34,35,36,37,38,39, 40,41,42,43,-1,44,45,46,
    47,48,49,50,51,52,53,54, 55,56,57,-1,-1,-1,-1,-1,
]
sha=lambda message:sha256(sha256(message).digest()).digest()

def encode_base58(v):
    if isinstance(v, str):
        v = v.encode('utf8')
    origlen = len(v)    
    v = v.lstrip(b'\0') # Skip & count leading zeroes.
    string = bytearray()
    acc=0
    for each in v:
        acc=(acc<<8)+each
    # acc = int(binascii.hexlify(v),16) # big-endian,if v=b'ab' then acc=0b0110000101100010
    while acc:
        acc, idx = divmod(acc, 58)
        string.insert(0,alphabet[idx])
    return b'1' * (origlen-len(v)) + string

def decode_base58(v):
    origlen = len(v)
    v = v.lstrip(b'1')
    acc = 0
    for char in v:
        acc = acc * 58 + mapping[char]
    result = bytearray()
    while acc:
        acc, mod = divmod(acc, 256)
        result.append(mod)
    return (b'\0' * (origlen - len(v)) + result[::-1])

def encode_check(v):
    if isinstance(v, str):
        v = v.encode('utf8')
    return encode_base58(v + sha(v)[:4])

def decode_check(v):
    v = decode_base58(v)
    result, check = v[:-4], v[-4:]
    if check != sha(result)[:4]:
        raise ValueError("Invalid checksum")
    return result

if __name__ == '__main__':
    print(encode_base58('hello world'))            # b'StV1DL6CwTryKyV'
    print(decode_base58(b'StV1DL6CwTryKyV'))       # b'hello world'
    print(encode_check('hello world'))             # b'3vQB7B6MrGQZaxCuFg4oh'
    print(decode_check(b'3vQB7B6MrGQZaxCuFg4oh'))  # b'hello world'
    # decode_check(b'4vQB7B6MrGQZaxCuFg4oh')       # ValueError: Invalid checksum
