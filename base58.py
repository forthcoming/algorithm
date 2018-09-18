'''
Implementations of Base58 and Base58Check endcodings that are compatible with the bitcoin network.
经过Base58编码的数据为原始的数据长度的8/log(2,58)倍,稍稍多于Base64的8/6=1.33倍
'''

from hashlib import sha256

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
    zeroes=origlen-len(v)
    string = bytearray(b'1'*zeroes)
    acc=0
    for each in v:
        acc=acc<<8)|each
    # acc = int(binascii.hexlify(v),16) # big-endian,if v=b'ab' then acc=0b0110000101100010
    while acc:
        acc, idx = divmod(acc, 58)
        string.insert(zeroes,alphabet[idx])
    return string

def decode_base58(v):
    ones=0
    while v[ones]==49: # '1'的ascii码
        ones+=1
    acc = 0
    for char in v[ones:]:
        acc = acc * 58 + mapping[char]
    result = bytearray()
    while acc:
        acc, mod = divmod(acc, 256)
        result.insert(0,mod)
    return b'\0' * ones + result

def _encode_base58(v):
    if isinstance(v, str):
        v = v.encode('utf8')
    origlen = len(v)    
    v = v.lstrip(b'\0') # Skip & count leading zeroes.
    newlen=len(v) 
    size=(newlen*138//100+1) # log(256) / log(58), rounded up.Allocate enough space in big-endian base58 representation.
    buffer=[0]*size
    length=0
    for carry in v:
        right=size-1
        _length=0
        while carry or _length<length:
            carry+=buffer[right]<<8
            buffer[right]=carry%58
            carry//=58
            right-=1
            _length+=1
        length=_length
    string = bytearray(b'1'*(origlen-newlen))
    for idx in buffer[right+1:]:
        string.append(alphabet[idx])
    return string

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
    print(encode_base58('hello world'))                    # bytearray(b'StV1DL6CwTryKyV')
    print(_encode_base58('hello world'))                   # bytearray(b'StV1DL6CwTryKyV')
    print(decode_base58(bytearray(b'StV1DL6CwTryKyV')))    # b'hello world'
    print(encode_check('akatsuki'))                        # bytearray(b'2qdLm9BNJAkjpnXz3')
    print(decode_check(bytearray(b'2qdLm9BNJAkjpnXz3')))   # b'akatsuki'
    # print(decode_check(bytearray(b'3qdLm9BNJAkjpnXz3'))) # ValueError: Invalid checksum
