#include<stdio.h>
#include<stdint.h>

uint32_t rotl32 ( uint32_t x, int8_t r )
{
  return (x << r) | (x >> (32 - r));
}

uint32_t fmix32 ( uint32_t h )
{
  h ^= h >> 16;
  h *= 0x85ebca6b;
  h ^= h >> 13;
  h *= 0xc2b2ae35;
  h ^= h >> 16;
  return h;
}

uint32_t MurmurHash3_x86_32 ( const void * key, int len, uint32_t seed)
{
  const uint8_t * data = (const uint8_t*)key;
  const int nblocks = len >>2;
  uint32_t h1 = seed;
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;

  const uint32_t * blocks = (const uint32_t *)(data + nblocks*4);
  for(int i = -nblocks; i; i++)
  {
    uint32_t k1 = blocks[i];

    k1 *= c1;
    k1 = rotl32(k1,15);
    k1 *= c2;

    h1 ^= k1;
    h1 = rotl32(h1,13);
    h1 = h1*5+0xe6546b64;
  }

  uint8_t * tail = (const uint8_t*)(data + nblocks*4);
  uint32_t k1 = 0;

  switch(len & 3)
  {
  case 3: 
    k1 ^= tail[2] << 16;
  case 2: 
    k1 ^= tail[1] << 8;
  case 1: 
    k1 ^= tail[0];
    k1 *= c1; 
    k1 = rotl32(k1,15); 
    k1 *= c2; 
    h1 ^= k1;
  };

  h1 ^= len;
  return fmix32(h1);
}

uint64_t MurmurHash64A ( const void * key, int len, unsigned int seed ) // 64-bit hash for 64-bit platforms,void *指向任意类型
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

int main()
{
  char a[3]="fax";
  uint64_t v1=MurmurHash64A(a,3,0x0);
  printf("%llx\n",v1);
        
  char str[11]="qwertyuiop";
  uint32_t v2=MurmurHash3_x86_32 (str,10 , 0x0);
  printf("%u\n",v2);
}
