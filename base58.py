#Implementations of Base58 and Base58Check endcodings that are compatible with the bitcoin network.

from hashlib import sha256
import binascii

alphabet = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'  # 58 character alphabet used

def b58encode(v):
    v = v.encode('utf8')
    zeros = len(v)
    v = v.lstrip(b'\0') # Skip & count leading zeroes.
    zeros -= len(v)
    acc = int(binascii.hexlify(v),16) # if v=b'ab' then acc=0b0110000101100010
    string = b""
    while acc:
        acc, idx = divmod(acc, 58)
        string = alphabet[idx:idx+1] + string
    return alphabet[0:1] * zeros + string

def b58decode(v):
    origlen = len(v)
    v = v.lstrip(alphabet[0:1])
    newlen = len(v)
    acc = 0
    for char in v:
        acc = acc * 58 + alphabet.index(char)
    result = []
    while acc > 0:
        acc, mod = divmod(acc, 256)
        result.append(mod)
    return (b'\0' * (origlen - newlen) + bytes(reversed(result)))

def b58encode_check(v):
    '''Encode a string using Base58 with a 4 character checksum'''
    digest = sha256(sha256(v).digest()).digest()
    return b58encode(v + digest[:4])

def b58decode_check(v):
    '''Decode and verify the checksum of a Base58 encoded string'''
    result = b58decode(v)
    result, check = result[:-4], result[-4:]
    digest = sha256(sha256(result).digest()).digest()
    if check != digest[:4]:
        raise ValueError("Invalid checksum")
    return result

if __name__ == '__main__':
    print(b58encode('hello world'))
    print(b58decode(b'StV1DL6CwTryKyV'))
