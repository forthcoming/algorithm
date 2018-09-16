# Implementations of Base58 and Base58Check endcodings that are compatible with the bitcoin network.

from hashlib import sha256
import binascii

alphabet = b'123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def encode_base58(v):
    if isinstance(v, str):
        v = v.encode('utf8')
    origlen = len(v)    
    v = v.lstrip(b'\0') # Skip & count leading zeroes.
    acc = int(binascii.hexlify(v),16) # if v=b'ab' then acc=0b0110000101100010
    string = b""
    while acc:
        acc, idx = divmod(acc, 58)
        string = alphabet[idx:idx+1] + string
    return alphabet[0:1] * (origlen-len(v)) + string

def decode_base58(v):
    origlen = len(v)
    v = v.lstrip(alphabet[0:1])
    acc = 0
    for char in v:
        acc = acc * 58 + alphabet.index(char)
    result = []
    while acc > 0:
        acc, mod = divmod(acc, 256)
        result.append(mod)
    return (b'\0' * (origlen - len(v)) + bytes(result[::-1]))

def encode_check(v):
    if isinstance(v, str):
        v = v.encode('utf8')
    digest = sha256(sha256(v).digest()).digest()
    return encode_base58(v + digest[:4])

def decode_check(v):
    v = decode_base58(v)
    result, check = v[:-4], v[-4:]
    digest = sha256(sha256(result).digest()).digest()
    if check != digest[:4]:
        raise ValueError("Invalid checksum")
    return result

if __name__ == '__main__':
    print(encode_base58('hello world'))            # b'StV1DL6CwTryKyV'
    print(decode_base58(b'StV1DL6CwTryKyV'))       # b'hello world'
    print(encode_check('hello world'))             # b'3vQB7B6MrGQZaxCuFg4oh'
    print(decode_check(b'3vQB7B6MrGQZaxCuFg4oh'))  # b'hello world'
    # decode_check(b'4vQB7B6MrGQZaxCuFg4oh')       # ValueError: Invalid checksum
