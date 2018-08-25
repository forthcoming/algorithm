#include<stdio.h>
#include<stdint.h>
#include<math.h>

#define HLL_BITS 6 // Enough to count up to 63 leading zeroes.
#define HLL_REGISTER_MAX ((1<<HLL_BITS)-1)
#define HLL_P 14 // The greater is P, the smaller the error.
#define HLL_Q (64-HLL_P) // The number of bits of the hash value used for determining the number of leading zeros.
#define HLL_REGISTERS (1<<HLL_P) // With P=14, 16384 registers.
#define HLL_P_MASK (HLL_REGISTERS-1) // Mask to index register.
#define HLL_ALPHA_INF 0.721347520444481703680 /* constant for 0.5/ln(2) */
#define HLL_DENSE 0 /* Dense encoding. */

typedef struct {
    char magic[4];      /* "HYLL" */
    uint8_t encoding;   /* HLL_DENSE or HLL_SPARSE. */
    uint8_t notused[3]; /* Reserved for future use, must be zero. */
    uint8_t card[8];    /* Cached cardinality, little endian. */
    uint8_t registers[]; /* Data bytes. */
} hllhdr;
/*
The use of 16384 6-bit registers for a great level of accuracy, using a total of 12k per key.
hllSigma,hllTau,hllCount三个函数没看懂,有时间再研究
Note: if we access the last counter, we will also access the b+1 byte
that is out of the array, but sds strings always have an implicit null
term, so the byte exists, and we can skip the conditional (or the need
to allocate 1 byte more explicitly).
*/
#define HLL_DENSE_GET_REGISTER(target,p,regnum) do { \
    uint8_t *_p = (uint8_t*) p; \
    unsigned long _byte = regnum*HLL_BITS/8; \
    unsigned long _fb = regnum*HLL_BITS&7; \
    unsigned long _fb8 = 8 - _fb; \
    unsigned long b0 = _p[_byte]; \
    unsigned long b1 = _p[_byte+1]; \
    target = ((b0 >> _fb) | (b1 << _fb8)) & HLL_REGISTER_MAX; \
} while(0)

#define HLL_DENSE_SET_REGISTER(p,regnum,val) do { \
    uint8_t *_p = (uint8_t*) p; \
    unsigned long _byte = regnum*HLL_BITS/8; \
    unsigned long _fb = regnum*HLL_BITS&7; \
    unsigned long _fb8 = 8 - _fb; \
    unsigned long _v = val; \
    _p[_byte] &= ~(HLL_REGISTER_MAX << _fb); \
    _p[_byte] |= _v << _fb; \
    _p[_byte+1] &= ~(HLL_REGISTER_MAX >> _fb8); \
    _p[_byte+1] |= _v >> _fb8; \
} while(0)

uint64_t MurmurHash64A( const void * key, int len, unsigned int seed ) // 64-bit hash for 64-bit platforms,void *指向任意类型
{
    const uint64_t m = 0xc6a4a7935bd1e995;
    const int r = 47;
    uint64_t h = seed ^ (len * m);
    const uint64_t * data = (const uint64_t *)key;  //  const int* a  表示*a是常量
    const uint64_t * end = data + (len>>3);         //  int* const a  表示a是常量

    while(data != end)
    {
        uint64_t k = *data++;
        k *= m;
        k ^= k >> r;
        k *= m;
        h ^= k;
        h *= m;
    }
    const unsigned char * data2 = (const unsigned char*)data;

    switch(len & 7)
    {  // pay attention,no break here,deal with all the remain elements
        case 7:
            h ^= (uint64_t)data2[6] << 48;
        case 6:
            h ^= (uint64_t)data2[5] << 40;
        case 5:
            h ^= (uint64_t)data2[4] << 32;
        case 4:
            h ^= (uint64_t)data2[3] << 24;
        case 3:
            h ^= (uint64_t)data2[2] << 16;
        case 2:
            h ^= (uint64_t)data2[1] << 8;
        case 1:
            h ^= (uint64_t)data2[0];
            h *= m;
    };

    h ^= h >> r;
    h *= m;
    h ^= h >> r;
    return h;
}

int hllPatLen(unsigned char *ele, size_t elesize, long *regp) {
    uint64_t hash, index;
    hash = MurmurHash64A(ele,elesize,0xadc83b19ULL);
    index = hash & HLL_P_MASK; // Register index.
    hash >>= HLL_P; // Remove bits used to address the register.
    hash |= ((uint64_t)1<<HLL_Q); // Make sure the loop terminates and count will be <= Q+1.
    int count=0;
    while((hash>>count&1)==0){
        ++count;
    }
    *regp = (int) index;
    return ++count;  // ++ here since we count the "00000...1" pattern.
}

int hllDenseAdd(uint8_t *registers, unsigned char *ele, size_t elesize) {
    long index;
    uint8_t count = hllPatLen(ele,elesize,&index);
    uint8_t oldcount;
    HLL_DENSE_GET_REGISTER(oldcount,registers,index);
    if (count > oldcount) {
        HLL_DENSE_SET_REGISTER(registers,index,count);
        return 1;
    }
    else {
        return 0;
    } 
}

// Compute the register histogram in the dense representation. 
void hllDenseRegHisto(uint8_t *registers, int* reghisto) {
    int j;
    if (HLL_REGISTERS == 16384 && HLL_BITS == 6) {
        uint8_t *r = registers;
        unsigned long r0, r1, r2, r3, r4, r5, r6, r7, r8, r9,r10, r11, r12, r13, r14, r15;
        for (j = 0; j < 1024; j++) {  // Handle 16 registers per iteration.
            r0 = r[0] & 0b111111;
            r1 = (r[0] >> 6 | r[1] << 2) & 0b111111;
            r2 = (r[1] >> 4 | r[2] << 4) & 0b111111;
            r3 = (r[2] >> 2) & 0b111111;
            r4 = r[3] & 0b111111;
            r5 = (r[3] >> 6 | r[4] << 2) & 0b111111;
            r6 = (r[4] >> 4 | r[5] << 4) & 0b111111;
            r7 = (r[5] >> 2) & 0b111111;
            r8 = r[6] & 0b111111;
            r9 = (r[6] >> 6 | r[7] << 2) & 0b111111;
            r10 = (r[7] >> 4 | r[8] << 4) & 0b111111;
            r11 = (r[8] >> 2) & 0b111111;
            r12 = r[9] & 0b111111;
            r13 = (r[9] >> 6 | r[10] << 2) & 0b111111;
            r14 = (r[10] >> 4 | r[11] << 4) & 0b111111;
            r15 = (r[11] >> 2) & 0b111111;

            reghisto[r0]++;
            reghisto[r1]++;
            reghisto[r2]++;
            reghisto[r3]++;
            reghisto[r4]++;
            reghisto[r5]++;
            reghisto[r6]++;
            reghisto[r7]++;
            reghisto[r8]++;
            reghisto[r9]++;
            reghisto[r10]++;
            reghisto[r11]++;
            reghisto[r12]++;
            reghisto[r13]++;
            reghisto[r14]++;
            reghisto[r15]++;
            r += 12;
        }
    }
    else {
        for(j = 0; j < HLL_REGISTERS; j++) {
            unsigned long reg;
            HLL_DENSE_GET_REGISTER(reg,registers,j);
            reghisto[reg]++;
        }
    }
}

double hllSigma(double x) {  // 0<x<1
    if (x == 1.) return INFINITY;
    double zPrime;
    double y = 1;
    double z = x;
    do {
        x *= x;
        zPrime = z;
        z += x * y;
        y += y;
    } while(zPrime != z);
    return z;
}

double hllTau(double x) { // 0<x<1
    if (x == 0. || x == 1.) return 0.;
    double zPrime;
    double y = 1.0;
    double z = 1 - x;
    do {
        x = sqrt(x);
        zPrime = z;
        y *= 0.5;
        z -= pow(1 - x, 2)*y;
    } while(zPrime != z);
    return z / 3;
}

uint64_t hllCount(hllhdr *hdr, int *invalid) {
    double m = HLL_REGISTERS;
    double E;
    int j;
    int reghisto[HLL_Q+2] = {0};
    if (hdr->encoding == HLL_DENSE) {
        hllDenseRegHisto(hdr->registers,reghisto);
    } 
    else {
        printf("Unknown HyperLogLog encoding in hllCount()");
    }
    double z = m * hllTau((m-reghisto[HLL_Q+1])/(double)m);
    for (j = HLL_Q; j >= 1; --j) {
        z += reghisto[j];
        z *= 0.5;
    }
    z += m * hllSigma(reghisto[0]/(double)m);
    E = llroundl(HLL_ALPHA_INF*m*m/z);
    return (uint64_t) E;
}

int hllMerge(uint8_t *max, hllhdr *hdr) {
    if (hdr->encoding == HLL_DENSE) {
        uint8_t val;
        for (int i = 0; i < HLL_REGISTERS; i++) {
            HLL_DENSE_GET_REGISTER(val,hdr->registers,i);
            if (val > max[i]) max[i] = val;
        }
        return 1;
    }
    return 0;
}

int main()
{
    char str[3]="fax";
    uint64_t result=MurmurHash64A(str,3,0xadc83b19ULL);
    printf("%llx\n",result);  // 13fb26a624822b1c  1001111111011001001101010011000100100100000100010101100011100

    uint8_t p[]={97,98,99,100,101,102,103,104};  // 1000011001000110110001100010011010100110011001101110011000010110
    uint8_t oldcount;
    HLL_DENSE_GET_REGISTER(oldcount,p,8);
    printf("%d\n",oldcount);  // 39

    long regp;
    int number=hllPatLen("fax",3,&regp);
    printf("%d\t%d\n",regp,number);  // 11036 4
}
