#include <stdio.h>
#include <stdint.h>
#include <math.h>

#define HASHISZERO(r) (!(r).bits && !(r).step)
#define GEO_STEP_MAX 26 /* 26*2 = 52 bits. */
#define GZERO(s) s.bits = s.step = 0;
#define D_R (M_PI / 180.0)

const double EARTH_RADIUS_IN_METERS = 6372797.560856;    // Earth's quatratic mean radius for WGS-84
const double MERCATOR_MAX = 20037726.37;                 // lat_range.max

typedef struct {
    uint64_t bits;
    uint8_t step;
} GeoHashBits;

typedef struct {
    double min;
    double max;
} GeoHashRange;

typedef struct {
    GeoHashBits hash;
    GeoHashRange longitude;
    GeoHashRange latitude;
} GeoHashArea;

typedef struct {
    GeoHashBits north;
    GeoHashBits east;
    GeoHashBits west;
    GeoHashBits south;
    GeoHashBits north_east;
    GeoHashBits south_east;
    GeoHashBits north_west;
    GeoHashBits south_west;
} GeoHashNeighbors;

typedef uint64_t GeoHashFix52Bits;

typedef struct {
    GeoHashBits hash;
    GeoHashArea area;
    GeoHashNeighbors neighbors;
} GeoHashRadius;

/**
 * Hashing works like this:
 * Divide the world into 4 buckets.  Label each one as such:
 *  -----------------
 *  |       |       |
 *  |       |       |
 *  | 0,1   | 1,1   |
 *  -----------------
 *  |       |       |
 *  |       |       |
 *  | 0,0   | 1,0   |
 *  -----------------
所有未加static前缀的全局变量和函数都具有全局可见性,其它的源文件也能访问,访问前需通过extern关键字声明
静态函数和静态全局变量作用域仅限于所在的源文件,可以在不同的文件中定义同名函数和同名变量而不必担心命名冲突
静态局部变量只初始化一次,存放在静态存储区,所以它具备持久性和默认值0(全局变量也存储在静态数据区,在静态数据区内存中所有的字节默认值都是0x00)
同时编译多个文件方法:gcc file1.c file2.c -o run
*/

static inline uint64_t interleave64(uint32_t xlo, uint32_t ylo) {
    /*
    Interleave lower bits of x and y, so the bits of x are in the even positions and bits from y in the odd(奇数);
    x and y must initially be less than 2**32 (65536). From: https://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
    0x5555555555555555ULL=0b0101010101010101010101010101010101010101010101010101010101010101
    0x3333333333333333ULL=0b0011001100110011001100110011001100110011001100110011001100110011
    0x0F0F0F0F0F0F0F0FULL=0b0000111100001111000011110000111100001111000011110000111100001111
    0x00FF00FF00FF00FFULL=0b0000000011111111000000001111111100000000111111110000000011111111
    0x0000FFFF0000FFFFULL=0b0000000000000000111111111111111100000000000000001111111111111111
    */
    static const uint64_t B[] = {0x5555555555555555, 0x3333333333333333,0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF,0x0000FFFF0000FFFF};
    static const unsigned int S[] = {1, 2, 4, 8, 16};
    uint64_t x = xlo;
    uint64_t y = ylo;

    x = (x | (x << S[4])) & B[4];
    y = (y | (y << S[4])) & B[4];

    x = (x | (x << S[3])) & B[3];
    y = (y | (y << S[3])) & B[3];

    x = (x | (x << S[2])) & B[2];
    y = (y | (y << S[2])) & B[2];

    x = (x | (x << S[1])) & B[1];
    y = (y | (y << S[1])) & B[1];

    x = (x | (x << S[0])) & B[0];
    y = (y | (y << S[0])) & B[0];

    return x | (y << 1);
    /*
    if x=x1,x2,...x32 and y=y1,y2,...y32
    then x=0,x1,0,x2,...0,x32 and y=0,y1,0,y2,...0,y32
    return y1,x1,y2,x2,...y32,x32
    */
}

static inline uint64_t deinterleave64(uint64_t interleaved) {
    /*
    reverse the interleave process
    if interleave=y1,x1,y2,x2,...y32,x32 then return y1,y2,...y32,x1,x2,...x32
    */
    static const uint64_t B[] = {0x5555555555555555, 0x3333333333333333,0x0F0F0F0F0F0F0F0F, 0x00FF00FF00FF00FF,0x0000FFFF0000FFFF, 0x00000000FFFFFFFF};
    static const unsigned int S[] = {0, 1, 2, 4, 8, 16};

    uint64_t x = interleaved;
    uint64_t y = interleaved >> 1;

    x = (x | (x >> S[0])) & B[0];
    y = (y | (y >> S[0])) & B[0];

    x = (x | (x >> S[1])) & B[1];
    y = (y | (y >> S[1])) & B[1];

    x = (x | (x >> S[2])) & B[2];
    y = (y | (y >> S[2])) & B[2];

    x = (x | (x >> S[3])) & B[3];
    y = (y | (y >> S[3])) & B[3];

    x = (x | (x >> S[4])) & B[4];
    y = (y | (y >> S[4])) & B[4];

    x = (x | (x >> S[5])) & B[5];
    y = (y | (y >> S[5])) & B[5];

    return x | (y << 32);
}

// inline只适合代码简单的函数,不能包含while、switch等复杂操作,并且内联函数本身不能是递归函数,如果函数体内的代码比较长,使用内联将导致内存消耗代价较高。
static inline double deg_rad(double ang) { return ang * D_R; }  // degree => radian
static inline double rad_deg(double ang) { return ang / D_R; }  // radian => degree

void geohashGetCoordRange(GeoHashRange *long_range, GeoHashRange *lat_range) {
    long_range->max = 180;
    long_range->min = -180;
    lat_range->max = 85.05112878;
    lat_range->min = -85.05112878;
}

int geohashEncode(const GeoHashRange *long_range, const GeoHashRange *lat_range,double longitude, double latitude, uint8_t step,GeoHashBits *hash) {
    if (hash == NULL || step > 32 || step == 0 || lat_range == NULL || long_range == NULL ||
        latitude < lat_range->min || latitude > lat_range->max || longitude < long_range->min || longitude > long_range->max
    ) return 0;

    hash->bits = 0;
    hash->step = step;
    double lat_offset =(latitude - lat_range->min) / (lat_range->max - lat_range->min);
    double long_offset =(longitude - long_range->min) / (long_range->max - long_range->min);
    /*
    eg:
    longitude=100,(long_range->min,long_range->max)=(-180,180)
    1. longitude ∈ (0,180)         左边占了.5    0.1
    2. longitude ∈ (90,180)        左边占了1*0.5+1*0.5^2    0.11
    3. longitude ∈ (90,135)        左边占了1*0.5+1*0.5^2+0*0.5^3    0.110
    4. longitude ∈ (90,112.5)      左边占了1*0.5+1*0.5^2+0*0.5^3+0*0.5^4    0.1100
    5. longitude ∈ (90,101.25)     左边占了1*0.5+1*0.5^2+0*0.5^3+0*0.5^4+0*0.5^5    0.11000
    6. longitude ∈ (95.625,101.25) 左边占了1*0.5+1*0.5^2+0*0.5^3+0*0.5^4+0*0.5^5+1*0.5^6    0.110001
    ......
    long_offset =(longitude - long_range->min) / (long_range->max - long_range->min)=0.110001110001......
    */
    lat_offset *= (1ULL << step); // 必须要加ULL,应为当step=31or32时符号位会变成负数or1直接溢出使结果变为0
    long_offset *= (1ULL << step);
    hash->bits = interleave64(lat_offset, long_offset);  // 隐式转换:double => uint32_t
    return 1;
}

int geohashEncodeWGS84(double longitude, double latitude, uint8_t step,GeoHashBits *hash) {
    GeoHashRange r[2] = {{0}};   // GeoHashRange r[2]={{.max=0,.min=1},{.min=2,.max=3}};
    geohashGetCoordRange(&r[0], &r[1]);
    return geohashEncode(&r[0], &r[1], longitude, latitude, step, hash);
}

int geohashDecode(const GeoHashRange long_range, const GeoHashRange lat_range,const GeoHashBits hash, GeoHashArea *area) {
    if (HASHISZERO(hash) || NULL == area || long_range.max || long_range.min || lat_range.max || lat_range.min)
        return 0;
    area->hash = hash;
    uint8_t step = hash.step;
    uint64_t hash_sep = deinterleave64(hash.bits); /* hash = [LAT][LONG] */

    double lat_scale = lat_range.max - lat_range.min;
    double long_scale = long_range.max - long_range.min;

    uint32_t ilato = hash_sep;       /* get lat part of deinterleaved hash */
    uint32_t ilono = hash_sep >> 32; /* shift over to get long part of hash */

    /* divide by 2**step.Then, for 0-1 coordinate, multiply times scale and add to the min to get the absolute coordinate. */
    area->latitude.min =lat_range.min + (ilato * 1.0 / (1ull << step)) * lat_scale;
    area->latitude.max =lat_range.min + ((ilato + 1) * 1.0 / (1ull << step)) * lat_scale;
    area->longitude.min =long_range.min + (ilono * 1.0 / (1ull << step)) * long_scale;
    area->longitude.max =long_range.min + ((ilono + 1) * 1.0 / (1ull << step)) * long_scale;
    return 1;
}

int geohashDecodeToLongLatWGS84(const GeoHashBits hash, double *xy) {
    GeoHashArea area = {{0}};
    GeoHashRange r[2] = {{0}};
    geohashGetCoordRange(&r[0], &r[1]);
    if (!xy || !geohashDecode(r[0], r[1], hash, &area))
        return 0;
    xy[0] = (area.longitude.min + area.longitude.max) / 2;
    xy[1] = (area.latitude.min + area.latitude.max) / 2;
    return 1;
}

static void geohash_move_x(GeoHashBits *hash, int8_t d) {  // 更改经度
    if (d == 0)
        return;
    uint64_t x;
    uint64_t y = hash->bits & 0x5555555555555555ULL; // 0b0101010101010101010101010101010101010101010101010101010101010101
    if (d > 0)
        x = (hash->bits | 0x5555555555555555ULL) + 1; // 使经度位二进制数加1(参考geohashEncode中的计算模型),超出长度step则x=0,相当于从180走到了-180
    else
        x = (hash->bits & 0xaaaaaaaaaaaaaaaaULL) - 1;// 0b1010101010101010101010101010101010101010101010101010101010101010
    x &= (0xaaaaaaaaaaaaaaaaULL >> (64 - hash->step * 2));
    hash->bits = (x | y);
}

static void geohash_move_y(GeoHashBits *hash, int8_t d) {
    if (d == 0)
        return;
    uint64_t x = hash->bits & 0xaaaaaaaaaaaaaaaaULL;
    uint64_t y ;
    if (d > 0)
        y = (hash->bits | 0xaaaaaaaaaaaaaaaaULL) + 1;
    else
        y = (hash->bits & 0x5555555555555555ULL) - 1;
    y &= (0x5555555555555555ULL >> (64 - hash->step * 2));
    hash->bits = (x | y);
}

void geohashNeighbors(const GeoHashBits *hash, GeoHashNeighbors *neighbors) {
    neighbors->east = *hash;    // 类型相同的结构体之间可以直接赋值
    neighbors->west = *hash;
    neighbors->north = *hash;
    neighbors->south = *hash;
    neighbors->south_east = *hash;
    neighbors->south_west = *hash;
    neighbors->north_east = *hash;
    neighbors->north_west = *hash;

    geohash_move_x(&neighbors->east, 1);  // -1,1是根据直角坐标系及方向共同确定
    geohash_move_x(&neighbors->west, -1);
    geohash_move_y(&neighbors->south, -1);
    geohash_move_y(&neighbors->north, 1);

    geohash_move_x(&neighbors->north_west, -1);
    geohash_move_y(&neighbors->north_west, 1);
    geohash_move_x(&neighbors->north_east, 1);
    geohash_move_y(&neighbors->north_east, 1);
    geohash_move_x(&neighbors->south_east, 1);
    geohash_move_y(&neighbors->south_east, -1);
    geohash_move_x(&neighbors->south_west, -1);
    geohash_move_y(&neighbors->south_west, -1);
}

// 计算地球两点间弧长距离,using haversin great circle distance formula.
double geohashGetDistance(double lon1d, double lat1d, double lon2d, double lat2d) {
    /*
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
    */
    double lat1r, lon1r, lat2r, lon2r, u, v;
    lat1r = deg_rad(lat1d);
    lon1r = deg_rad(lon1d);
    lat2r = deg_rad(lat2d);
    lon2r = deg_rad(lon2d);
    u = sin((lat2r - lat1r) / 2);  // 三角函数参数是弧度而非角度
    v = sin((lon2r - lon1r) / 2);
    return 2.0 * EARTH_RADIUS_IN_METERS * asin(sqrt(u * u + cos(lat1r) * cos(lat2r) * v * v));
}

// This function is used in order to estimate the step (bits precision) of the 9 search area boxes during radius queries.
uint8_t geohashEstimateStepsByRadius(double range_meters, double lat) {
    if (range_meters == 0) return 26;  // 按指定搜索半径估计geohashEncode所需步长,最大不超过26
    uint8_t step = 1;
    while (range_meters < MERCATOR_MAX) {
        range_meters *= 2;
        ++step;
    }
    step -= 2; /* Make sure range is included in most of the base cases. */

    /* Wider range torwards the poles... Note: it is possible to do better
     * than this approximation by computing the distance between meridians
     * at this latitude, but this does the trick for now. */
    if (lat > 66 || lat < -66) {
        --step;
        if (lat > 80 || lat < -80) 
            --step;
    }

    /* Frame to valid range. */
    if (step < 1) step = 1;
    if (step > 26) step = 26;
    return step;
}

/* Return the bounding box of the search area centered at latitude,longitude
 * having a radius of radius_meter. bounds[0] - bounds[2] is the minimum
 * and maxium longitude, while bounds[1] - bounds[3] is the minimum and
 * maximum latitude.
 *
 * This function does not behave correctly with very large radius values, for
 * instance for the coordinates 81.634948934258375 30.561509253718668 and a
 * radius of 7083 kilometers, it reports as bounding boxes:
 *
 * min_lon 7.680495, min_lat -33.119473, max_lon 155.589402, max_lat 94.242491
 *
 * However, for instance, a min_lon of 7.680495 is not correct, because the
 * point -1.27579540014266968 61.33421815228281559 is at less than 7000
 * kilometers away.
 *
 * Since this function is currently only used as an optimization, the
 * optimization is not used for very big radiuses, however the function
 * should be fixed. */
int geohashBoundingBox(double longitude, double latitude, double radius_meters, double *bounds) {
    if (!bounds) return 0;

    bounds[0] = longitude - rad_deg(radius_meters/EARTH_RADIUS_IN_METERS/cos(deg_rad(latitude))); // EARTH_RADIUS_IN_METERS*cos(deg_rad(latitude)):维度所在的平行赤道的圆半径
    bounds[2] = longitude + rad_deg(radius_meters/EARTH_RADIUS_IN_METERS/cos(deg_rad(latitude)));
    bounds[1] = latitude - rad_deg(radius_meters/EARTH_RADIUS_IN_METERS);
    bounds[3] = latitude + rad_deg(radius_meters/EARTH_RADIUS_IN_METERS);
    return 1;
}

/* Return a set of areas (center + 8) that are able to cover a range query for the specified position and radius. */
GeoHashRadius geohashGetAreasByRadiusWGS84(double longitude, double latitude, double radius_meters) {
    GeoHashRange long_range, lat_range;
    GeoHashRadius radius;
    GeoHashBits hash;
    GeoHashNeighbors neighbors;
    GeoHashArea area;
    double min_lon, max_lon, min_lat, max_lat;
    double bounds[4];
    uint8_t step=geohashEstimateStepsByRadius(radius_meters,latitude);

    geohashBoundingBox(longitude, latitude, radius_meters, bounds);
    min_lon = bounds[0];
    min_lat = bounds[1];
    max_lon = bounds[2];
    max_lat = bounds[3];
    
    geohashGetCoordRange(&long_range,&lat_range);
    geohashEncode(&long_range,&lat_range,longitude,latitude,step,&hash);
    geohashNeighbors(&hash,&neighbors);

    /*
    Check if the step is enough at the limits of the covered area.Sometimes when the search area is near an edge of the area,
    the estimated step is not small enough, since one of the north / south / west / east square is too near to the search area to cover everything.
    */
    int decrease_step = 0;
    { 
        GeoHashArea north, south, east, west;  // 大括号内部的变量仅仅用于检测,不想被外面访问

        geohashDecode(long_range, lat_range, neighbors.north, &north);
        geohashDecode(long_range, lat_range, neighbors.south, &south);
        geohashDecode(long_range, lat_range, neighbors.east, &east);
        geohashDecode(long_range, lat_range, neighbors.west, &west);

        if (geohashGetDistance(longitude,latitude,longitude,north.latitude.max)< radius_meters ||
            geohashGetDistance(longitude,latitude,longitude,south.latitude.min)< radius_meters ||
            geohashGetDistance(longitude,latitude,east.longitude.max,latitude)< radius_meters ||
            geohashGetDistance(longitude,latitude,west.longitude.min,latitude)< radius_meters
        ) decrease_step = 1;
    }

    if (step > 1 && decrease_step) {  // 保证相临的八个点(其实只需要东南西北四个点)落在搜索半径之外
        --step;
        geohashEncode(&long_range,&lat_range,longitude,latitude,step,&hash);
        geohashNeighbors(&hash,&neighbors);
    }
   
    geohashDecode(long_range,lat_range,hash,&area);
    if (step >= 2) {  // Exclude the search areas that are useless.
        if (area.latitude.min < min_lat) {
            GZERO(neighbors.south);
            GZERO(neighbors.south_west);
            GZERO(neighbors.south_east);
        }
        if (area.latitude.max > max_lat) {
            GZERO(neighbors.north);
            GZERO(neighbors.north_east);
            GZERO(neighbors.north_west);
        }
        if (area.longitude.min < min_lon) {
            GZERO(neighbors.west);
            GZERO(neighbors.south_west);
            GZERO(neighbors.north_west);
        }
        if (area.longitude.max > max_lon) {
            GZERO(neighbors.east);
            GZERO(neighbors.south_east);
            GZERO(neighbors.north_east);
        }
    }
    radius.hash = hash;
    radius.neighbors = neighbors;
    radius.area = area;
    return radius;
}

GeoHashFix52Bits geohashAlign52Bits(const GeoHashBits hash) {
    uint64_t bits = hash.bits;
    bits <<= (52 - hash.step * 2);
    return bits;
}

int geohashGetDistanceIfInRadiusWGS84(double x1, double y1, double x2, double y2, double radius, double *distance) {
    *distance = geohashGetDistance(x1, y1, x2, y2);
    if (*distance > radius) return 0;
    return 1;
}

int main(){
    uint32_t x=0;
    uint32_t y=0b11111111111111111111111111111111;
    printf("%llu\n",deinterleave64(interleave64(x, y)));

    GeoHashBits hash={ .bits=1234567,.step=20};
    GeoHashNeighbors neighbors;
    geohashNeighbors(&hash,&neighbors);
}
