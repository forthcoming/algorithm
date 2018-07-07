#include<stdio.h>
#include<stdint.h>

uint64_t MurmurHash64A ( const void * key, int len, unsigned int seed ) // 64-bit hash for 64-bit platforms,void *指向任意类型
{
        const uint64_t m = 0xc6a4a7935bd1e995;
        const int r = 47;
        uint64_t h = seed ^ (len * m);
        const uint64_t * data = (const uint64_t *)key;  //  const int* a  表示*a是常量
        const uint64_t * end = data + (len>>3);         //  int* const a  表示a是常量

        while(data != end){
            uint64_t k = *data++;
            k *= m;
            k ^= k >> r;
            k *= m;
            h ^= k;
            h *= m;
        }
        const unsigned char * data2 = (const unsigned char*)data;

        switch(len & 7){  // pay attention,no break here,deal with all the remain elements
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

int main(){
    char a[3]="fax";
    uint64_t x=MurmurHash64A(a,sizeof(a)/sizeof(char),0x0);
    printf("%llx",x);
}
