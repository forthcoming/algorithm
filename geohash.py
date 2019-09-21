__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'
__mapping = {letter:index for index,letter in enumerate(__base32)}

def geo_encode(longitude, latitude, geo_length=11):
    assert -180<=longitude<=180 and -85.05112878<=latitude<=85.05112878
    lon_interval,lat_interval = (-180.0, 180.0),(-85.05112878,85.05112878)  # 与redis保持一致
    geohash = 0
    _geohash=[]
    even=True
    for i in range(geo_length):
        for j in [16,8,4,2,1]:
            if even:
                mid = (lon_interval[0] + lon_interval[1]) / 2   # 偶数位放经度
                if longitude > mid:
                    geohash|=j
                    lon_interval = (mid, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], mid)
            else:
                mid = (lat_interval[0] + lat_interval[1]) / 2
                if latitude > mid:
                    geohash|=j
                    lat_interval = (mid, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], mid)
            even=not even
        _geohash.append(__base32[geohash])
        geohash=0
    return ''.join(_geohash)  # 字符串越长,表示的范围越精确,字符串相似表示距离相近,可以利用字符串的前缀匹配来查询附近信息

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


class GEO:
    _earth_radius = 6372797.560856

    def __init__(self,step=30):
        assert 0<step<=32
        self.scale=1<<step

    def encode(self,longitude,latitude):
        assert -180<=longitude<=180 and -90<=latitude<=90
        lon_offset = longitude/360 + .5
        lat_offset = latitude/180 + .5
        lon_offset *= self.scale
        lat_offset *= self.scale
        return GEO.interleave64(int(lat_offset),int(lon_offset))
        # return __class__.interleave64(int(lat_offset),int(lon_offset))  # py3

    def decode(self,geohash):
        # lon_offset,lat_offset = __class__.deinterleave64(geohash)
        lon_offset,lat_offset = GEO.deinterleave64(geohash)
        longitude = 360*lon_offset/self.scale-180
        latitude = 180*lat_offset/self.scale-90
        return longitude,latitude 

    @staticmethod
    def distance(lon1, lat1, lon2, lat2):
        percent = math.pi / 180
        lon1 *= percent
        lat1 *= percent
        lon2 *= percent
        lat2 *= percent
        u = math.sin((lat2 - lat1) / 2)
        v = math.sin((lon2 - lon1) / 2)
        # return round(2 * __class__._earth_radius * math.asin((u * u + math.cos(lat1) * math.cos(lat2) * v * v)**.5))
        return round(2 * GEO._earth_radius * math.asin((u * u + math.cos(lat1) * math.cos(lat2) * v * v)**.5))

    @staticmethod
    def interleave64(x, y):
        B = [0x5555555555555555, 0x3333333333333333,0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF,0x0000FFFF0000FFFF]
        S = [1, 2, 4, 8, 16]

        x = (x | (x << S[4])) & B[4]  
        y = (y | (y << S[4])) & B[4] 
    
        x = (x | (x << S[3])) & B[3] 
        y = (y | (y << S[3])) & B[3] 
    
        x = (x | (x << S[2])) & B[2] 
        y = (y | (y << S[2])) & B[2] 
    
        x = (x | (x << S[1])) & B[1] 
        y = (y | (y << S[1])) & B[1] 
    
        x = (x | (x << S[0])) & B[0] 
        y = (y | (y << S[0])) & B[0] 
    
        return x | (y << 1) 

    @staticmethod
    def deinterleave64(interleaved):
        B = [0x3333333333333333,0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF,0x0000FFFF0000FFFF, 0x00000000FFFFFFFF]
        S = [1, 2, 4, 8, 16]

        x = interleaved & 0x5555555555555555
        y = (interleaved >> 1) & 0x5555555555555555
    
        x = (x | (x >> S[0])) & B[0]
        y = (y | (y >> S[0])) & B[0]
    
        x = (x | (x >> S[1])) & B[1]
        y = (y | (y >> S[1])) & B[1]
    
        x = (x | (x >> S[2])) & B[2]
        y = (y | (y >> S[2])) & B[2]
    
        x = (x | (x >> S[3])) & B[3]
        y = (y | (y >> S[3])) & B[3]
    
        x = (x | (x >> S[4])) & B[4]
        y = (y | (y >> S[4])) & B[4]
        
        return y, x
 
