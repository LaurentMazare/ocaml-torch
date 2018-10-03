#ifndef __TORCH_API_H__
#define __TORCH_API_H__

#ifdef __cplusplus
extern "C" {
typedef at::Tensor *tensor;
#else
typedef void *tensor;
#endif

tensor at_zeros(int *, int, int);
tensor at_ones(int *, int, int);
tensor at_rand(int *, int);
tensor at_reshape(tensor, int *, int);

int at_dim(tensor);
void at_shape(tensor, int *);
int at_scalar_type(tensor);

tensor at_add(tensor, tensor);
tensor at_sub(tensor, tensor);
tensor at_mul(tensor, tensor);
tensor at_div(tensor, tensor);
tensor at_pow(tensor, tensor);
tensor at_matmul(tensor, tensor);

void at_print(tensor);

void at_free(tensor);

#ifdef __cplusplus
};
#endif

#endif
