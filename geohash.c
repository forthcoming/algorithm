#include <stdio.h>
#include <stdint.h>

#define HASHISZERO(r) (!(r).bits && !(r).step)
#define RANGEISZERO(r) (!(r).max && !(r).min)
#define RANGEPISZERO(r) (r == NULL || RANGEISZERO(*r))
#define GEO_STEP_MAX 26 /* 26*2 = 52 bits. */
#define GZERO(s) s.bits = s.step = 0;

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

void geohashGetCoordRange(GeoHashRange *long_range, GeoHashRange *lat_range) {
    long_range->max = 180;
    long_range->min = -180;
    lat_range->max = 85.05112878;
    lat_range->min = -85.05112878;
}

int geohashEncode(const GeoHashRange *long_range, const GeoHashRange *lat_range,double longitude, double latitude, uint8_t step,GeoHashBits *hash) {
    // long_range地址非空,并且long_range.max和long_range.min都不能为0,类似的还有lat_range
    // TODO:这里判断需要精简
    if (hash == NULL || step > 32 || step == 0 || RANGEPISZERO(lat_range) || RANGEPISZERO(long_range) ||
        longitude > 180 || longitude < -180 || latitude > 85.05112878 || latitude < -85.05112878 ||
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
    if (HASHISZERO(hash) || NULL == area || RANGEISZERO(lat_range) || RANGEISZERO(long_range))
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
        x = (hash->bits | 0x5555555555555555ULL) + 1; // 使经度位二进制数加1,超出长度step则x=0
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

int main(){
    uint32_t x=0;
    uint32_t y=0b11111111111111111111111111111111;
    printf("%llu\n",deinterleave64(interleave64(x, y)));
    double lat_offset = .8;
    lat_offset *= (1ULL << 10);
    printf("%u\n",(uint32_t)lat_offset);
}
