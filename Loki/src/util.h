#ifndef UTIL_H
#define UTIL_H

#include <sys/time.h>
//#include <loki/lokilib.h>


/*
 * Get a current timestamp with us accuracy. This will give you the time that
 * has passed since a certain point in time. While the value itself doesn't
 * tell you much, you can subtract timestamps from each other to get the
 * amount of time that has passed between them.
 */
const int MAX_POOLING_LAYER_NUM_CORE = 8;
const int CONV_LAYER_NUM_CORE = 8;

static inline uint64_t timestamp_us() {
  struct timeval tv;
  gettimeofday(&tv,NULL);
  return 1000000L * tv.tv_sec + tv.tv_usec;
}


int is_little_endian(){
  int x = 1;
  char *y = (char*)&x;
  printf("%c\n",*y+48);
  return *y+48;
}

#endif
