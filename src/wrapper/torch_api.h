#ifndef __TORCH_API_H__
#define __TORCH_API_H__
#include<stdint.h>

#ifdef __cplusplus
extern "C" {
typedef torch::Tensor *tensor;
#else
typedef void *tensor;
#endif

tensor at_float_vec(double *values, int value_len, int type);
tensor at_int_vec(int64_t *values, int value_len, int type);
tensor at_zeros(int *dims, int dim_len, int type);
tensor at_ones(int *dims, int dim_len, int type);
tensor at_rand(int *dims, int dim_len);
tensor at_reshape(tensor, int *dims, int dim_len);

int at_dim(tensor);
void at_shape(tensor, int *);
int at_scalar_type(tensor);

tensor at_sum(tensor);
tensor at_mean(tensor);
tensor at_argmax(tensor);
tensor at_softmax(tensor);
tensor at_neg(tensor);
tensor at_log(tensor);
tensor at_add(tensor, tensor);
tensor at_sub(tensor, tensor);
tensor at_mul(tensor, tensor);
tensor at_div(tensor, tensor);
tensor at_pow(tensor, tensor);
tensor at_matmul(tensor, tensor);
tensor at_eq(tensor, tensor);

void at_sub_assign(tensor, tensor);

void at_backward(tensor);
tensor at_grad(tensor);
tensor at_set_requires_grad(tensor, int);

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

#ifdef __cplusplus
};
#endif

#endif
