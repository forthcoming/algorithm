def fmix( h ):
    h ^= h >> 16
    h  = ( h * 0x85ebca6b ) & 0xFFFFFFFF
    h ^= h >> 13
    h  = ( h * 0xc2b2ae35 ) & 0xFFFFFFFF
    h ^= h >> 16
    return h

# https://github.com/forthcoming/algorithm/blob/master/MurmurHash3.c
# https://github.com/wc-duck/pymmh3/blob/master/pymmh3.py#L34  
def murmur_hash( key, seed ):  # Implements MurmurHash3_x86_32 hash.
    key = bytearray(key.encode())  # pay attention
    length = len( key )
    nblocks = length >>2
    h1 = seed

    c1 = 0xcc9e2d51
    c2 = 0x1b873593

    for block_start in range( 0, nblocks * 4, 4 ):
        k1 = key[ block_start + 3 ] << 24 | key[ block_start + 2 ] << 16 |  key[ block_start + 1 ] <<  8 |  key[ block_start ]  # little endian           
        k1 = ( c1 * k1 ) & 0xFFFFFFFF
        k1 = ( k1 << 15 | k1 >> 17 ) & 0xFFFFFFFF # inlined ROTL32
        k1 = ( c2 * k1 ) & 0xFFFFFFFF
        
        h1 ^= k1
        h1  = ( h1 << 13 | h1 >> 19 ) & 0xFFFFFFFF # inlined ROTL32
        h1  = ( h1 * 5 + 0xe6546b64 ) & 0xFFFFFFFF

    tail_index = nblocks * 4
    k1 = 0
    tail_size = length & 3

    if tail_size >= 3:
        k1 ^= key[ tail_index + 2 ] << 16
    if tail_size >= 2:
        k1 ^= key[ tail_index + 1 ] << 8
    if tail_size >= 1:
        k1 ^= key[ tail_index + 0 ]
    if tail_size > 0:
        k1  = ( k1 * c1 ) & 0xFFFFFFFF
        k1  = ( k1 << 15 | k1 >> 17 ) & 0xFFFFFFFF # inlined ROTL32
        k1  = ( k1 * c2 ) & 0xFFFFFFFF
        h1 ^= k1
    return fmix( h1 ^ length )

# https://github.com/antirez/redis/blob/unstable/src/hyperloglog.c#L339
def hll_dense_get_register(p,regnum):
    hll_bits=6
    p=bytearray(p.encode())
    binary=''.join('{:08b}'.format(each)[::-1] for each in p)
    # print(binary)
    left=regnum*hll_bits
    return int(binary[left:left+hll_bits][::-1],2)

if __name__ == "__main__":
    print(murmur_hash('qwertyuiop',0x0))
    print(hll_dense_get_register('abcdefgh',8))
