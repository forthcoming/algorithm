alphabet=b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"

def encode_base32(message):  # base64.b32encode(b'abcd')
    length=len(message)
    idx=0
    buffer=0  # c里面直接用int类型表示buffer即可
    left=0
    result=bytearray()  
    while left>5 or idx<length:
        if left<5:
            buffer=(buffer<<8) | message[idx]
            left+=8
            idx+=1
        result.append(alphabet[0b11111 & (buffer>>(left-5))])
        left-=5
    if left: # 防止出现空串
        result.append(alphabet[0b11111 & (buffer<<(5-left))])
    pad=len(result)%8
    if pad:
        result.extend([61]*(8-pad)) # 长度必须是8的整倍数,不足的用=填充
    return result
    
def decode_base32(message):
    length=len(message)
    idx=0
    buffer=0
    left=0
    result=bytearray()  
    while idx<length:
        ch=message[idx]
        if 65<=ch<=90:   # [A,Z]
            ch=(ch&0b11111)-1
        elif 50<=ch<=55: # [2,7]
            ch-=24
        else:
            break
        idx+=1
        buffer=(buffer<<5)|ch
        left+=5
        if left>=8:
            result.append(0b11111111&(buffer>>(left-8)))
            left-=8
    return result
    
