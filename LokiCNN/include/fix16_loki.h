#include <stdint.h>
typedef int32_t fix16_t;


static const fix16_t fix16_maximum  = 0x7FFFFFFF; /*!< the maximum value of fix16_t */
static const fix16_t fix16_minimum  = 0x80000000; /*!< the minimum value of fix16_t */
static const fix16_t fix16_overflow = 0x80000000; /*!< the value used to indicate overflows when FIXMATH_NO_OVERFLOW is not specified */
static const fix16_t fix16_one = 0x00010000; /*!< fix16_t value of 1 */
static const fix16_t fix16_e   = 178145;     /*!< fix16_t value of e */

static inline fix16_t fix16_from_int(int a)     { return a * fix16_one; }
static inline float   fix16_to_float(fix16_t a) { return (float)a / fix16_one; }
static inline double  fix16_to_dbl(fix16_t a)   { return (double)a / fix16_one; }

static inline int fix16_to_int(fix16_t a)
{
#ifdef FIXMATH_NO_ROUNDING
    return (a >> 16);
#else
	if (a >= 0)
		return (a + (fix16_one >> 1)) / fix16_one;
	return (a - (fix16_one >> 1)) / fix16_one;
#endif
}

static inline fix16_t fix16_from_float(float a)
{
	float temp = a * fix16_one;
#ifndef FIXMATH_NO_ROUNDING
	temp += (temp >= 0) ? 0.5f : -0.5f;
#endif
	return (fix16_t)temp;
}

static inline fix16_t fix16_from_dbl(double a)
{
	double temp = a * fix16_one;
#ifndef FIXMATH_NO_ROUNDING
	temp += (temp >= 0) ? 0.5f : -0.5f;
#endif
	return (fix16_t)temp;
}

static inline fix16_t multiply(fix16_t i1, fix16_t i2){
  int64_t product = (int64_t)i1 * i2;
  fix16_t result = product >> 16;
  #if defined(FIX16_ROUNDING)
    result += (product && 0x8000) >> 15;
  #endif
  return result;
}

static inline fix16_t add(fix16_t i1, fix16_t i2){
  return i1 + i2;
}


static inline fix16_t add_multiply(fix16_t s1, fix16_t m1, fix16_t m2){
  int64_t product = (int64_t)m1 * m2;
  fix16_t result = product >> 16;
  result = result + s1;
  return result;
}

static inline fix16_t readDouble(double input) {return fix16_from_dbl(input);}
static inline fix16_t readInt(int input) {return fix16_from_int(input);}
static inline fix16_t divide(fix16_t d1, fix16_t d2){
	float x = fix16_to_float(d1);
	float y = fix16_to_float(d2);
	float z = x / y;
	return fix16_from_float(z);
}

static inline fix16_t exp_t(fix16_t x){
	float in_float = fix16_to_float(x);
	float z = exp(in_float);
	return fix16_from_float(z);
}