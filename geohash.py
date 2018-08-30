__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
__mapping = {letter:index for index,letter in enumerate(__base32)}

def geo_encode(longitude, latitude, geolength=11):
    assert -180<=longitude<=180 and -85.05112878<=latitude<=85.05112878
    lon_interval,lat_interval = (-180.0, 180.0),(-90.0, 90.0)
    geohash = 0
    even=True
    for i in range(geolength*5):
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2   # 偶数位放经度
            if longitude > mid:
                geohash=(geohash<<1)|1
                lon_interval = (mid, lon_interval[1])
            else:
                geohash<<=1
                lon_interval = (lon_interval[0], mid)
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                geohash=(geohash<<1)|1
                lat_interval = (mid, lat_interval[1])
            else:
                geohash<<=1
                lat_interval = (lat_interval[0], mid)
        even=not even
    return ''.join(__base32[geohash>>i&0b11111] for i in range(geolength*5-5,-5,-5))  # 字符串越长,表示的范围越精确,字符串相似表示距离相近,可以利用字符串的前缀匹配来查询附近信息

def geo_decode(geohash):
    lon_interval,lat_interval = (-180.0, 180.0),(-90.0, 90.0)
    even = True
    for letter in geohash:
        index = __mapping[letter]
        for mask in [16, 8, 4, 2, 1]:
            if even:
                if index & mask:
                    lon_interval = ((lon_interval[0]+lon_interval[1])/2, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], (lon_interval[0]+lon_interval[1])/2)
            else:
                if index & mask:
                    lat_interval = ((lat_interval[0]+lat_interval[1])/2, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], (lat_interval[0]+lat_interval[1])/2)
            even = not even
    return (lon_interval[0] + lon_interval[1]) / 2,(lat_interval[0] + lat_interval[1]) / 2
