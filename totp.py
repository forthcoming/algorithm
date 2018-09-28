# designed to work for one-time passwords - HMAC-based and time-based. It is compatible with Google Authenticator application and applications based on it.

import base64,hashlib,hmac,struct,time

def _is_possible_token(token, token_length=6):
    token = str(token)
    return token.isdigit() and len(token) <= token_length

def hotp(secret,intervals_no,digest_method=hashlib.sha1,token_length=6):
    """
    secret: the base32-encoded string acting as secret key
    intervals_no: interval number used for getting different tokens, it is incremented with each use
    """
    key = base64.b32decode(secret)
    msg = struct.pack('>Q', intervals_no)
    hmac_digest = hmac.new(key, msg, digest_method).digest()
    o = hmac_digest[19] & 0b1111
    token_base = struct.unpack('>I', hmac_digest[o:o + 4])[0] & 0x7fffffff
    token = token_base % (10 ** token_length)
    return token


def totp(secret,digest_method=hashlib.sha1,token_length=6,interval_length=30,clock=None):
    if clock is None:
        clock=int(time.time())
    interv_no = clock // interval_length
    return hotp(secret,intervals_no=interv_no,digest_method=digest_method,token_length=token_length)

def valid_totp(token,secret,digest_method=hashlib.sha1,token_length=6,interval_length=30,window=0):
    """
    Check if given token is valid time-based one-time password for given secret.
    window: compensate for clock skew, number of intervals to check on each side of the current time. (default is 0 - only check the current clock time)
    """
    if _is_possible_token(token, token_length=token_length):
        for w in range(-window, window+1):
            if token == totp(secret,digest_method=digest_method,token_length=token_length,interval_length=interval_length,clock=int(time.time())+(w*interval_length)):
                return True
    return False

if __name__=='__main__':
    secret = b'MFRGGZDFMZTWQ2LK'
    token = totp(secret)
    print(valid_totp(token, secret))   # True
    print(valid_totp(token+1, secret)) # False
