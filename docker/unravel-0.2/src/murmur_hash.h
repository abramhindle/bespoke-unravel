#ifndef _MURMUR_HASH_H_
#define _MURMUR_HASH_H_

//NOTE: quite fast, nice collision properties, but endian dependent hash values

#include "have_64_bits.h"
typedef uintptr_t MurmurInt;

// MurmurHash2, by Austin Appleby

static const uint32_t DEFAULT_SEED=2654435769U;

#if HAVE_64_BITS
//MurmurInt MurmurHash(void const *key, int len, uint32_t seed=DEFAULT_SEED);

inline uint64_t MurmurHash64( const void * key, int len, unsigned int seed=DEFAULT_SEED )
{
  const uint64_t m = 0xc6a4a7935bd1e995ULL;
  const int r = 47;

  uint64_t h = seed ^ (len * m);

  const uint64_t * data = (const uint64_t *)key;
  const uint64_t * end = data + (len/8);

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
  {
  case 7: h ^= uint64_t(data2[6]) << 48;
  case 6: h ^= uint64_t(data2[5]) << 40;
  case 5: h ^= uint64_t(data2[4]) << 32;
  case 4: h ^= uint64_t(data2[3]) << 24;
  case 3: h ^= uint64_t(data2[2]) << 16;
  case 2: h ^= uint64_t(data2[1]) << 8;
  case 1: h ^= uint64_t(data2[0]);
    h *= m;
  };

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

inline uint32_t MurmurHash32(void const *key, int len, uint32_t seed=DEFAULT_SEED)
{
  return (uint32_t) MurmurHash64(key,len,seed);
}

inline MurmurInt MurmurHash(void const *key, int len, uint32_t seed=DEFAULT_SEED)
{
  return MurmurHash64(key,len,seed);
}

#else
// 32-bit

// Note - This code makes a few assumptions about how your machine behaves -
// 1. We can read a 4-byte value from any address without crashing
// 2. sizeof(int) == 4
inline uint32_t MurmurHash32 ( const void * key, int len, uint32_t seed=DEFAULT_SEED)
{
  // 'm' and 'r' are mixing constants generated offline.
  // They're not really 'magic', they just happen to work well.

  const uint32_t m = 0x5bd1e995;
  const int r = 24;

  // Initialize the hash to a 'random' value

  uint32_t h = seed ^ len;

  // Mix 4 bytes at a time into the hash

  const unsigned char * data = (const unsigned char *)key;

  while(len >= 4)
  {
    uint32_t k = *(uint32_t *)data;

    k *= m;
    k ^= k >> r;
    k *= m;

    h *= m;
    h ^= k;

    data += 4;
    len -= 4;
  }

  // Handle the last few bytes of the input array

  switch(len)
  {
  case 3: h ^= data[2] << 16;
  case 2: h ^= data[1] << 8;
  case 1: h ^= data[0];
    h *= m;
  };

  // Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;
}

inline MurmurInt MurmurHash ( const void * key, int len, uint32_t seed=DEFAULT_SEED) {
  return MurmurHash32(key,len,seed);
}

// 64-bit hash for 32-bit platforms

inline uint64_t MurmurHash64 ( const void * key, int len, uint32_t seed=DEFAULT_SEED)
{
  const uint32_t m = 0x5bd1e995;
  const int r = 24;

  uint32_t h1 = seed ^ len;
  uint32_t h2 = 0;

  const uint32_t * data = (const uint32_t *)key;

  while(len >= 8)
  {
    uint32_t k1 = *data++;
    k1 *= m; k1 ^= k1 >> r; k1 *= m;
    h1 *= m; h1 ^= k1;
    len -= 4;

    uint32_t k2 = *data++;
    k2 *= m; k2 ^= k2 >> r; k2 *= m;
    h2 *= m; h2 ^= k2;
    len -= 4;
  }

  if(len >= 4)
  {
    uint32_t k1 = *data++;
    k1 *= m; k1 ^= k1 >> r; k1 *= m;
    h1 *= m; h1 ^= k1;
    len -= 4;
  }

  switch(len)
  {
  case 3: h2 ^= ((unsigned char*)data)[2] << 16;
  case 2: h2 ^= ((unsigned char*)data)[1] << 8;
  case 1: h2 ^= ((unsigned char*)data)[0];
    h2 *= m;
  };

  h1 ^= h2 >> 18; h1 *= m;
  h2 ^= h1 >> 22; h2 *= m;
  h1 ^= h2 >> 17; h1 *= m;
  h2 ^= h1 >> 19; h2 *= m;

  uint64_t h = h1;

  h = (h << 32) | h2;

  return h;
}

#endif
//32bit

inline uint64_t MurmurHash64( const std::string& str ) {
  return MurmurHash64(str.c_str(), strlen(str.c_str()) );
}

inline uint64_t MurmurHash32( const std::string& str ) {
  return MurmurHash32(str.c_str(), strlen(str.c_str()) );
}

#endif
