#include <stdint.h>
typedef int32_t fix8_t;

#define WIDTH_OF_FLOATS 10
// static const fix16_t fix16_maximum  = 0x7FFFFFFF; /*!< the maximum value of fix16_t */
// static const fix16_t fix16_minimum  = 0x80000000; /*!< the minimum value of fix16_t */
// static const fix16_t fix16_overflow = 0x80000000; /*!< the value used to indicate overflows when FIXMATH_NO_OVERFLOW is not specified */
static const fix8_t fix8_one = (1 << WIDTH_OF_FLOATS); /*!< fix16_t value of 1 */


// static const fix16_t fix16_e   = 178145;     /*!< fix16_t value of e */

static inline float   fix8_to_float(fix8_t a) { return (float)a / fix8_one; }
static inline double  fix8_to_dbl(fix8_t a)   { return (double)a / fix8_one; }

static inline fix8_t read_from_int(int a) {return a*fix8_one;}

static inline int fix8_to_int(fix8_t a)
{
#ifdef FIXMATH_NO_ROUNDING
    return (a >> WIDTH_OF_FLOATS);
#else
	if (a >= 0)
		return (a + (fix8_one >> 1)) / fix8_one;
	return (a - (fix8_one >> 1)) / fix8_one;
#endif
}

static inline fix8_t fix8_from_float(float a)
{
	float temp = a * fix8_one;
#ifndef FIXMATH_NO_ROUNDING
	temp += (temp >= 0) ? 0.5f : -0.5f;
#endif
	return (fix8_t)temp;
}

static inline fix8_t fix8_from_dbl(double a)
{
	double temp = a * fix8_one;
#ifndef FIXMATH_NO_ROUNDING
	temp += (temp >= 0) ? 0.5f : -0.5f;
#endif
	return (fix8_t)temp;
}

static inline fix8_t divide(fix8_t d1, fix8_t d2){
	float x = fix8_to_float(d1);
	float y = fix8_to_float(d2);
	float z = x / y;
	return fix8_from_float(z);
}

static inline fix8_t exp_t(fix8_t x){
	float in_float = fix8_to_float(x);
	float z = exp(in_float);
	return fix8_from_float(z);
}

static inline fix8_t readDouble(double x){
	return fix8_from_dbl(x);
}

static inline fix8_t add(fix8_t x, fix8_t y){
	return x + y;
}

static inline fix8_t multiply(fix8_t x, fix8_t y){
	int32_t z = x * y;
	z = z >> WIDTH_OF_FLOATS;
	return (fix8_t)z;
}

static inline fix8_t add_multiply(fix8_t a, fix8_t x, fix8_t y){
	int32_t z = (int32_t)x * y;
	fix8_t result = z >> WIDTH_OF_FLOATS;
	result = result + a;
	return result;
}