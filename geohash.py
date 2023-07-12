import math


class GeoHash:
    earth_radius_in_meters = 6372797.560856  # Earth's quatratic mean radius for WGS-84
    geo_alphabet = "0123456789bcdefghjkmnpqrstuvwxyz"
    mapping = {letter: index for index, letter in enumerate(geo_alphabet)}
    max_lon, max_lat = 180, 90

    def __init__(self, step=26):
        assert 0 < step <= 32
        self.step = step
        self.geo_length = math.ceil(step * .4)  # step * 2 / 5

    def encode(self, longitude, latitude):
        """
        假设longitude=100
        1. longitude ∈ (0,180)         左边占总体0.5                                           = b0.1
        2. longitude ∈ (90,180)        左边占总体1*0.5+1*0.5^2                                 = b0.11
        3. longitude ∈ (90,135)        左边占总体1*0.5+1*0.5^2+0*0.5^3                         = b0.110
        4. longitude ∈ (90,112.5)      左边占总体1*0.5+1*0.5^2+0*0.5^3+0*0.5^4                 = b0.1100
        5. longitude ∈ (90,101.25)     左边占总体1*0.5+1*0.5^2+0*0.5^3+0*0.5^4+0*0.5^5         = b0.11000
        6. longitude ∈ (95.625,101.25) 左边占总体1*0.5+1*0.5^2+0*0.5^3+0*0.5^4+0*0.5^5+1*0.5^6 = b0.110001
        ......
        从上面可以看出最左边占总体比例long_offset的小数部分正好等于geohash中对精度的编码
        """
        assert longitude <= abs(GeoHash.max_lon) and latitude <= abs(GeoHash.max_lat)
        lon_offset = longitude / GeoHash.max_lon / 2 + .5
        lat_offset = latitude / GeoHash.max_lat / 2 + .5
        scale = 1 << self.step
        lon_offset *= scale
        lat_offset *= scale
        interleave = GeoHash.interleave64(int(lon_offset), int(lat_offset))
        # geohash = ""
        # double_step = self.step << 1
        # for i in range(self.geo_length - 1):
        #     idx = (interleave >> (double_step - ((i + 1) * 5))) & 0b11111
        #     geohash = f"{geohash}{GeoHash.geo_alphabet[idx]}"
        # geohash = f"{geohash}{GeoHash.geo_alphabet[0]}"
        # return geohash  # hash值与redis保持一致
        return interleave

    def decode(self, interleave):
        lon_offset, lat_offset = GeoHash.de_interleave64(interleave)
        scale = 1 << self.step
        longitude_min = 2 * GeoHash.max_lon * lon_offset / scale - GeoHash.max_lon
        longitude_max = 2 * GeoHash.max_lon * (lon_offset + 1) / scale - GeoHash.max_lon
        latitude_min = 2 * GeoHash.max_lat * lat_offset / scale - GeoHash.max_lat
        latitude_max = 2 * GeoHash.max_lat * (lat_offset + 1) / scale - GeoHash.max_lat
        longitude = (longitude_min + longitude_max) / 2
        if longitude > GeoHash.max_lon:
            longitude = GeoHash.max_lon
        if longitude < -GeoHash.max_lon:
            longitude = -GeoHash.max_lon
        latitude = (latitude_min + latitude_max) / 2
        if latitude > GeoHash.max_lat:
            latitude = GeoHash.max_lat
        if latitude < -GeoHash.max_lat:
            latitude = -GeoHash.max_lat
        return longitude, latitude

    @staticmethod
    def interleave64(x, y):
        """
        refer: https://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
        0x5555555555555555 = 0b0101010101010101010101010101010101010101010101010101010101010101
        0x3333333333333333 = 0b0011001100110011001100110011001100110011001100110011001100110011
        0x0F0F0F0F0F0F0F0F = 0b0000111100001111000011110000111100001111000011110000111100001111
        0x00FF00FF00FF00FF = 0b0000000011111111000000001111111100000000111111110000000011111111
        0x0000FFFF0000FFFF = 0b0000000000000000111111111111111100000000000000001111111111111111
        x和y初值必须小于2**32(uint32_t类型), 后面新的x,y是64位(uint64_t类型)
        if x=x1,x2,...x32 and y=y1,y2,...y32
            then x=0,x1,0,x2,...0,x32 and y=0,y1,0,y2,...0,y32
            return x1,y1,x2,y2,...x32,y32
        """
        b = [0x5555555555555555, 0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF, 0x0000FFFF0000FFFF]
        s = [1, 2, 4, 8, 16]

        x = (x | (x << s[4])) & b[4]
        y = (y | (y << s[4])) & b[4]

        x = (x | (x << s[3])) & b[3]
        y = (y | (y << s[3])) & b[3]

        x = (x | (x << s[2])) & b[2]
        y = (y | (y << s[2])) & b[2]

        x = (x | (x << s[1])) & b[1]
        y = (y | (y << s[1])) & b[1]

        x = (x | (x << s[0])) & b[0]
        y = (y | (y << s[0])) & b[0]

        return (x << 1) | y

    @staticmethod
    def de_interleave64(interleaved):
        b = [0x3333333333333333, 0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF, 0x0000FFFF0000FFFF, 0x00000000FFFFFFFF]
        s = [1, 2, 4, 8, 16]

        x = (interleaved >> 1) & 0x5555555555555555
        y = interleaved & 0x5555555555555555

        x = (x | (x >> s[0])) & b[0]
        y = (y | (y >> s[0])) & b[0]

        x = (x | (x >> s[1])) & b[1]
        y = (y | (y >> s[1])) & b[1]

        x = (x | (x >> s[2])) & b[2]
        y = (y | (y >> s[2])) & b[2]

        x = (x | (x >> s[3])) & b[3]
        y = (y | (y >> s[3])) & b[3]

        x = (x | (x >> s[4])) & b[4]
        y = (y | (y >> s[4])) & b[4]

        return x, y

    def move_x(self, interleaved, positive_direction=True):  # 更改经度
        x = interleaved & 0b1010101010101010101010101010101010101010101010101010101010101010
        y = interleaved & 0b0101010101010101010101010101010101010101010101010101010101010101
        zz = 0b0101010101010101010101010101010101010101010101010101010101010101 >> (64 - self.step * 2)
        if positive_direction:  # 使经度位二进制数加1(参考geohashEncode中的计算模型),超出长度step则x=0,相当于从180走到了-180
            x += zz + 1
        else:
            x |= zz
            x -= zz + 1
        x &= 0b1010101010101010101010101010101010101010101010101010101010101010 >> (64 - self.step * 2)
        return x | y

    def move_y(self, interleaved, positive_direction=True):
        x = interleaved & 0b1010101010101010101010101010101010101010101010101010101010101010
        y = interleaved & 0b0101010101010101010101010101010101010101010101010101010101010101
        zz = 0b1010101010101010101010101010101010101010101010101010101010101010 >> (64 - self.step * 2)
        if positive_direction:
            y += zz + 1
        else:
            y |= zz
            y -= zz + 1
        y &= 0b0101010101010101010101010101010101010101010101010101010101010101 >> (64 - self.step * 2)
        return x | y

    def neighbors(self, interleaved):  # 以某个点为中心范围查询时用到
        east = self.move_x(interleaved, True)
        west = self.move_x(interleaved, False)
        south = self.move_y(interleaved, False)
        north = self.move_y(interleaved, True)
        north_west = self.move_x(interleaved, False)
        north_west = self.move_y(north_west, True)
        north_east = self.move_x(interleaved, True)
        north_east = self.move_y(north_east, True)
        south_east = self.move_x(interleaved, True)
        south_east = self.move_y(south_east, False)
        south_west = self.move_x(interleaved, False)
        south_west = self.move_y(south_west, False)
        return [east, west, south, north, north_west, north_east, south_east, south_west]

    def encode_lower_version(self, longitude, latitude):
        assert longitude <= abs(GeoHash.max_lon) and latitude <= abs(GeoHash.max_lat)
        lon_interval = (-GeoHash.max_lon, GeoHash.max_lon)
        lat_interval = (-GeoHash.max_lat, GeoHash.max_lat)
        geohash = 0
        _geohash = ""
        even = True
        for i in range(self.geo_length):
            for j in [16, 8, 4, 2, 1]:
                if even:
                    mid = (lon_interval[0] + lon_interval[1]) / 2  # 偶数位放经度
                    if longitude > mid:
                        geohash |= j
                        lon_interval = (mid, lon_interval[1])
                    else:
                        lon_interval = (lon_interval[0], mid)
                else:
                    mid = (lat_interval[0] + lat_interval[1]) / 2
                    if latitude > mid:
                        geohash |= j
                        lat_interval = (mid, lat_interval[1])
                    else:
                        lat_interval = (lat_interval[0], mid)
                even = not even
            _geohash = f"{_geohash}{GeoHash.geo_alphabet[geohash]}"
            geohash = 0
        return _geohash  # 字符串越长,表示的范围越精确,字符串相似表示距离相近,可以利用字符串的前缀匹配来查询附近信息

    @staticmethod
    def decode_lower_version(geohash):
        lon_interval = (-GeoHash.max_lon, GeoHash.max_lon)
        lat_interval = (-GeoHash.max_lat, GeoHash.max_lat)
        even = True
        for letter in geohash:
            index = GeoHash.mapping[letter]
            for mask in [16, 8, 4, 2, 1]:
                if even:
                    mid = (lon_interval[0] + lon_interval[1]) / 2
                    if index & mask:
                        lon_interval = (mid, lon_interval[1])
                    else:
                        lon_interval = (lon_interval[0], mid)
                else:
                    mid = (lat_interval[0] + lat_interval[1]) / 2
                    if index & mask:
                        lat_interval = (mid, lat_interval[1])
                    else:
                        lat_interval = (lat_interval[0], mid)
                even = not even
        return sum(lon_interval) / 2, sum(lat_interval) / 2

    @staticmethod
    def distance(lon1, lat1, lon2, lat2):
        """
        计算地球两点间弧长距离,using Great-circle distance Haversine formula
        整体思想是把经纬度(球坐标系)转换成空间直角坐标系,利用三角函数计算两点间的弧度,从而计算出弧长
        a=(lo1,la1) and b=(lo2,la2)
        A=r(SIN(90-la1)COS(180-lo1),SIN(90-la1)SIN(180-lo1),COS(90-la1))
         =r(-COSla1COSlo1,COSla1SINlo1,SINla1)
        B=r(-COSla2COSlo2,COSla2SINlo2,SINla2)
        COS<aob=COSla1COSlo1*COSla2COSlo2+COSla1SINlo1*COSla2SINlo2+SINla1*SINla2
               =COSla1*COSla2*COS(lo2-lo1)+SINla1*SINla2
               =COSla1*COSla2*(1-2v^2)+SINla1*SINla2
               =COS(la2-la1)-2v^2*COSla1COSla2
               =1-2u^2-2v^2*COSla1COSla2
        <aob=2*asin(sqrt(u^2 + v^2*COSla1COSla2))  # 弧度
        """
        percent = math.pi / 180
        lon1 *= percent
        lon2 *= percent
        lat1 *= percent
        lat2 *= percent
        v = math.sin((lon2 - lon1) / 2)
        u = math.sin((lat2 - lat1) / 2)  # 三角函数参数是弧度而非角度
        a = u * u + math.cos(lat1) * math.cos(lat2) * v * v
        return round(2 * GeoHash.earth_radius_in_meters * math.asin(a ** .5), 4)


if __name__ == "__main__":
    lon, lat = 13.361389, 38.115556
    geo = GeoHash()
    encoded = geo.encode(lon, lat)
    decoded = geo.decode(encoded)
    print(encoded, decoded)
    for neighbor in geo.neighbors(geo.encode(lon, lat)):
        print(GeoHash.distance(lon, lat, *geo.decode(neighbor)))

    encoded = geo.encode_lower_version(lon, lat)
    decoded = geo.decode_lower_version(encoded)
    print(encoded, decoded)
