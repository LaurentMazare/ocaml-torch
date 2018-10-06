#ifndef __TORCH_API_H__
#define __TORCH_API_H__
#include<stdint.h>

#ifdef __cplusplus
extern "C" {
typedef torch::Tensor *tensor;
#define PROTECT(x) \
  try { \
    x \
  } catch (const exception& e) { \
    caml_failwith(strdup(e.what())); \
  }
#else
typedef void *tensor;
#endif

tensor at_float_vec(double *values, int value_len, int type);
tensor at_int_vec(int64_t *values, int value_len, int type);

int at_dim(tensor);
void at_shape(tensor, int *);
int at_scalar_type(tensor);

void at_backward(tensor);
tensor at_grad(tensor);
tensor at_set_requires_grad(tensor, int);
int at_requires_grad(tensor);

tensor at_get(tensor, int index);
tensor at_select(tensor, int dim, int index);
double at_double_value(tensor);
int64_t at_int64_value(tensor);
void at_fill_double(tensor, double);
void at_fill_int64(tensor, int64_t);

void at_set_double2(tensor, int dim1, int dim2, double value);

void at_print(tensor);
void at_save(tensor, char *filename);
tensor at_load(char *filename);

void at_free(tensor);

#include "torch_api_generated.h"

#ifdef __cplusplus
};
#endif

#endif
