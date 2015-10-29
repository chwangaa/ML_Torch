#ifndef UTIL_H
#define UTIL_H

#include <sys/time.h>

/*
 * Get a current timestamp with us accuracy. This will give you the time that
 * has passed since a certain point in time. While the value itself doesn't
 * tell you much, you can subtract timestamps from each other to get the
 * amount of time that has passed between them.
 */

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



#define debug_print(fmt, ...) \
            do { if (DEBUG)fprintf(stderr, fmt, __VA_ARGS__); } while (0)


#ifdef LOKI
#include <loki/lokilib.h>


#define function_summary(func, ...) \
            do { \
              if(DEBUG){ \
              unsigned long cycle_count = get_cycle_count(); \
              func(__VA_ARGS__); \
              debug_print(#func " takes %lu cycle to complete \n", get_cycle_count()-cycle_count); \
              } \
              else{ \
              func(__VA_ARGS__); \
              } \
            } while(0)
#else

#define function_summary(func, ...) \
            do { \
              if(DEBUG){ \
              uint64_t start_time = timestamp_us(); \
              func(__VA_ARGS__); \
              uint64_t end_time = timestamp_us(); \
              double dt = (double)(end_time-start_time) / 1000.0; \
              debug_print(#func " takes %lf ms to complete \n", dt); \
              } \
              else{ \
              func(__VA_ARGS__); \
              } \
            } while(0)

#endif

#ifdef LOKI
void loki_sync_simple(int cores){
  if (cores <= 1)
    return;

  uint core = get_core_id();

  // (after setting up a connection).
  if (core > 0) {
    loki_receive_token(3);
  } 
  else {
    // All core 0s then synchronise between tiles using the same process.
      int bitmask = all_cores_except_0(cores);
      int address = loki_mcast_address(bitmask, 3, false);
      set_channel_map(3, address);
      loki_send_token(3);
  }
}
#endif

#endif
