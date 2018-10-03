#ifndef __TORCH_API_H__
#define __TORCH_API_H__

#ifdef __cplusplus
extern "C" {
typedef at::Tensor *tensor;
#else
typedef void *tensor;
#endif

tensor at_zeros(int *, int);
tensor at_ones(int *, int);
tensor at_add(tensor, tensor);
void at_free(tensor);
void at_print(tensor);

#ifdef __cplusplus
};
#endif

#endif
